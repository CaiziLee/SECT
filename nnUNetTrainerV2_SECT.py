from collections import OrderedDict
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn, tensor
from torch.cuda.amp import autocast, GradScaler
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.consistency_loss import SoftmaxMSELoss
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from time import time
from nnunet.utilities.learning_utils import sigmoid_rampup
import sys
import torch.backends.cudnn as cudnn
from _warnings import warn
from tqdm import trange
from nnunet.utilities.tensor_utilities import sum_tensor
from torch.optim.lr_scheduler import _LRScheduler
from nnunet.training.dataloading.custom_variants.custom_dataset_loading import DataLoader2DWithIndexFixCrop, DataLoader2DWithIndex
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
import torch.nn.functional as F


class SoftmaxMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert target.requires_grad == False
        assert input.size() == target.size()
        input_softmax = torch.softmax(input, dim=1)
        target_softmax = torch.softmax(target, dim=1)
        return super().forward(input_softmax, target_softmax)


class DataLoader2DWithIndexFixCrop(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None, shuffle=True):
        """
        Created for pycontrast
        return slices and the corresponding index

        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With htis strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DWithIndexFixCrop, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.list_of_keys.sort()
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        # generate samples and corresponding index
        self.sample_index = {}
        self.num_samples = 0
        self.slices_num_list = []
        for j, i in enumerate(self.list_of_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            
            if j == 0:
                self.sample_index[i] = 0
            else:
                self.sample_index[i] = self.sample_index[self.list_of_keys[j - 1]] + num_slices
            
            num_slices = properties['size_after_resampling'][0]
            self.num_samples += num_slices
            self.slices_num_list.append(num_slices)

        self.sample_pointer = 0
        self.slice_pointer = 0

        self.shuffle = shuffle
    
    def __len__(self):
        return self.num_samples

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def generate_unshuffle_batchids(self):
        selected_keys = []
        slice_ids = []
        while len(slice_ids) < self.batch_size:
            selected_keys.append(self.list_of_keys[self.sample_pointer])
            slice_ids.append(self.slice_pointer)
            self.slice_pointer += 1
            if self.slice_pointer == self.slices_num_list[self.sample_pointer]:
                self.slice_pointer = 0
                self.sample_pointer += 1
                if self.sample_pointer == len(self.list_of_keys):
                    self.sample_pointer = 0
        return selected_keys, slice_ids
    
    def reset_pointers(self):
        self.sample_pointer = 0
        self.slice_pointer = 0

    def generate_train_batch(self):
        if self.shuffle:
            selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        else:
            selected_keys, slice_ids = self.generate_unshuffle_batchids()

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        case_index = []
        valid_bboxes = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            selected_class = None
            if self.shuffle:
                # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
                if not force_fg:
                    random_slice = np.random.choice(case_all_data.shape[1])
                    selected_class = None
                else:
                    # these values should have been precomputed
                    if 'class_locations' not in properties.keys():
                        raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                    foreground_classes = np.array(
                        [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                    foreground_classes = foreground_classes[foreground_classes > 0]
                    if len(foreground_classes) == 0:
                        selected_class = None
                        random_slice = np.random.choice(case_all_data.shape[1])
                        print('case does not contain any foreground classes', i)
                    else:
                        selected_class = np.random.choice(foreground_classes)

                        voxels_of_that_class = properties['class_locations'][selected_class]
                        valid_slices = np.unique(voxels_of_that_class[:, 0])
                        random_slice = np.random.choice(valid_slices)
                        voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                        voxels_of_that_class = voxels_of_that_class[:, 1:]
            else:
                random_slice = slice_ids[j]
            
            # infer the index of slices
            case_index.append(self.sample_index[i] + random_slice)

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:  # discard the random padding
                # bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                # bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_x_lb = (ub_x + 1 - lb_x) // 2
                bbox_y_lb = (ub_y + 1 - lb_y) // 2
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

            valid_bboxes.append({'id': case_index[-1], 'x_lb': valid_bbox_x_lb, 'x_ub': valid_bbox_x_ub, 'y_lb': valid_bbox_y_lb, 'y_ub': valid_bbox_y_ub})

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys, "index": case_index, "bbox": valid_bboxes}


class DataLoader2DWithIndex(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        Created for pycontrast
        return slices and the corresponding index

        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With htis strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DWithIndex, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.list_of_keys.sort()
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        # generate samples and corresponding index
        self.sample_index = {}
        self.num_samples = 0
        for j, i in enumerate(self.list_of_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            
            if j == 0:
                self.sample_index[i] = 0
            else:
                self.sample_index[i] = self.sample_index[self.list_of_keys[j - 1]] + num_slices
            
            num_slices = properties['size_after_resampling'][0]
            self.num_samples += num_slices
    
    def __len__(self):
        return self.num_samples

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        case_index = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]
            
            # infer the index of slices
            case_index.append(self.sample_index[i] + random_slice)

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys, "index": case_index}



class nnUNetTrainerV2_SECT(nnUNetTrainerV2):
    """
    custom_trainer_v1
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, num_training_cases='all'):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        self.initial_lr = 1e-2
        self.num_batches_per_epoch = 100
        self.num_val_batches_per_epoch = 20
        self.patience = 20

        self.save_every = 5    # save checkpoints
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

        # define number of training samples for label-effecient experiments, it will determine the data split during initialion
        self.num_training_cases = num_training_cases

        # super params for MT
        self.apply_consistency = True
        self.consistency = 0.1
        self.consistency_rampup = 40.0
        self.ema_decay = 0.99
        self.iter_step = 0

        self.apply_TE = True
        self.alpha = 0.6
        self.update_unlabeled_pseudo_labels = False

        self.all_tr_sup_losses = []
        self.all_tr_consistency_losses = []
        self.all_tr_te_losses = []
        self.all_val_sup_losses = []
        self.all_val_consistency_losses = []
        self.all_val_te_losses = []

        self.all_tr_losses1 = []
        self.all_tr_sup_losses1 = []
        self.all_tr_consistency_losses1 = []
        self.all_tr_te_losses1 = []
        self.all_val_losses1 = []
        self.all_val_sup_losses1 = []
        self.all_val_consistency_losses1 = []
        self.all_val_te_losses1 = []

        self.online_eval_foreground_dc1 = []
        self.online_eval_tp1 = []
        self.online_eval_fp1 = []
        self.online_eval_fn1 = []
        self.all_val_eval_metrics1 = []

        self.oversample_foreground_percent = 0.0

    def initialize(self, training=True, force_load_plans=False):
        """
        This is a copy from nnUNetTrainerV2 but the loss function is changed into "DC_and_CE_loss"
        
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function is changed into default(DC_and_CE_loss)

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            self.consistency_loss = SoftmaxMSELoss()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            
            if training:
                self.log_dir = join(self.output_folder, 'log_dir')
                if not isdir(self.log_dir):
                    makedirs(self.log_dir)
                self.summary_writer = SummaryWriter(log_dir=self.log_dir)

                self.dl_tr, self.dl_val, self.dl_tr_unlabeled, self.dl_tr_unlabeled_false_shuffle = self.get_basic_generators()
                self.num_labeled_data, self.num_val_data, self.num_unlabeled_data = len(self.dl_tr), len(self.dl_val), len(self.dl_tr_unlabeled)

                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )

                self.tr_gen_unlabeled, _ = get_no_augmentation(
                    self.dl_tr_unlabeled, self.dl_tr_unlabeled_false_shuffle,
                    params=self.data_aug_params,
                    pin_memory=self.pin_memory,
                )   # do not augment the unlabeled data in case of unconsistency during ensembling

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)

                self.targets_ensemble = torch.zeros((self.num_unlabeled_data, self.num_classes, *self.patch_size), dtype=torch.float32).cuda()
                self.targets_pseudo = torch.zeros((self.num_unlabeled_data, self.num_classes, *self.patch_size), dtype=torch.float32).cuda()
                self.targets_ensemble1 = torch.zeros((self.num_unlabeled_data, self.num_classes, *self.patch_size), dtype=torch.float32).cuda()
                self.targets_pseudo1 = torch.zeros((self.num_unlabeled_data, self.num_classes, *self.patch_size), dtype=torch.float32).cuda()
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

        self.print_to_log_file('nnUNetTrainerV2_SECT is initialized')
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader2DWithIndex(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2DWithIndex(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_unlabeled = DataLoader2DWithIndexFixCrop(self.dataset_tr_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2DWithIndex(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2DWithIndex(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_unlabeled = DataLoader2DWithIndexFixCrop(self.dataset_tr_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_unlabeled_false_shuffle = DataLoader2DWithIndexFixCrop(self.dataset_tr_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', shuffle=False)
        return dl_tr, dl_val, dl_tr_unlabeled, dl_tr_unlabeled_false_shuffle

    def initialize_network(self):
        """
        This is a copy from nnUNetTrainerV2 but the deep supervision is set to False
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = False 

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        assert self.network.do_ds is False, "deep_supervision is True"

        self.network1 = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        assert self.network1.do_ds is False, "deep_supervision is True"

        if torch.cuda.is_available():
            self.network.cuda()
            self.network1.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        self.network1.inference_apply_nonlin = softmax_helper
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, betas=(0.9, 0.999))
        self.lr_scheduler = None

        # self.optimizer1 = torch.optim.SGD(self.network1.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, betas=(0.9, 0.999))
        self.lr_scheduler1 = None

    def momentum_update(self, model, model_ema, m, step):
        m = min(1 - 1 / (step + 1), m)
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(p1.detach().data, alpha=1 - m)

    def run_online_evaluation_1(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))
    
    def run_online_evaluation_2(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc1.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp1.append(list(tp_hard))
            self.online_eval_fp1.append(list(fp_hard))
            self.online_eval_fn1.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        self.online_eval_tp1 = np.sum(self.online_eval_tp1, 0)
        self.online_eval_fp1 = np.sum(self.online_eval_fp1, 0)
        self.online_eval_fn1 = np.sum(self.online_eval_fn1, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        global_dc_per_class1 = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp1, self.online_eval_fp1, self.online_eval_fn1)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics1.append(np.mean(global_dc_per_class1))

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("Average global foreground Dice1:", [np.round(i, 4) for i in global_dc_per_class1])
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.online_eval_foreground_dc1 = []
        self.online_eval_tp1 = []
        self.online_eval_fp1 = []
        self.online_eval_fn1 = []

    def get_current_consistency_weight(self, weight):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return weight * sigmoid_rampup(self.epoch, self.consistency_rampup)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict1 = next(data_generator)
        data1 = data_dict1['data']
        target1 = data_dict1['target']

        # data_dict2 = next(data_generator)
        # data2 = data_dict2['data']
        # target2 = data_dict2['target']

        unlabeled_data_dict = next(self.tr_gen_unlabeled)
        unlabeled_data = unlabeled_data_dict['data']
        unlabeled_index = unlabeled_data_dict['index']

        data1 = maybe_to_torch(data1)
        target1 = maybe_to_torch(target1)
        # data2 = maybe_to_torch(data2)
        # target2 = maybe_to_torch(target2)
        unlabeled_data = maybe_to_torch(unlabeled_data)

        if torch.cuda.is_available():
            data1 = to_cuda(data1)
            target1 = to_cuda(target1)
            # data2 = to_cuda(data2)
            # target2 = to_cuda(target2)
            unlabeled_data = to_cuda(unlabeled_data)

        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()

        consistency_weight = self.get_current_consistency_weight(self.consistency)

        inputs = torch.cat((data1, unlabeled_data))
        # inputs1 = torch.cat((data2, unlabeled_data))

        if self.fp16:
            with autocast():
                # sup loss
                outputs = self.network(inputs)
                sup_loss = self.loss(outputs[: self.batch_size], target1)
                outputs1 = self.network1(inputs)
                sup_loss1 = self.loss(outputs1[: self.batch_size], target1)
                
                # pseudo loss
                consistency_loss = torch.tensor(0.0)
                consistency_loss1 = torch.tensor(0.0)
                if self.apply_consistency:
                    pseudo_logits, pseudo_label = torch.max(softmax_helper(outputs[self.batch_size:].detach()), dim=1)
                    pseudo_label = pseudo_label.unsqueeze(1).long()
                    pseudo_logits1, pseudo_label1 = torch.max(softmax_helper(outputs1[self.batch_size:].detach()), dim=1)
                    pseudo_label1 = pseudo_label1.unsqueeze(1).long()

                    consistency_loss = consistency_weight * self.loss(outputs[self.batch_size:], pseudo_label1.detach())
                    consistency_loss1 = consistency_weight * self.loss(outputs1[self.batch_size:], pseudo_label.detach())
                else:
                    consistency_loss = torch.tensor(0.0)
                    consistency_loss1 = torch.tensor(0.0)

                # TE
                te_loss = torch.tensor(0.0)
                te_loss1 = torch.tensor(0.0)
                if self.apply_TE:
                    outputs_ul_current = outputs[self.batch_size:]
                    outputs1_ul_current = outputs1[self.batch_size:]
                    
                    if self.epoch > 3:
                        temp_ensemble = self.targets_pseudo[unlabeled_index].clone()
                        temp_ensemble = self.alpha * temp_ensemble + (1 - self.alpha) * F.normalize(outputs_ul_current.detach(), p=2, dim=1)
                        te_loss = consistency_weight * self.consistency_loss(outputs_ul_current, temp_ensemble)

                        temp_ensemble1 = self.targets_pseudo1[unlabeled_index].clone()
                        temp_ensemble1 = self.alpha * temp_ensemble1 + (1 - self.alpha) * F.normalize(outputs1_ul_current.detach(), p=2, dim=1)
                        te_loss1 = consistency_weight * self.consistency_loss(outputs1_ul_current, temp_ensemble1)

                    if do_backprop is False and self.epoch > 2 and self.update_unlabeled_pseudo_labels and self.iter_step % self.num_batches_per_epoch == 0:   
                        self.dl_tr_unlabeled_false_shuffle.reset_pointers()
                        self.tr_gen_unlabeled_false_shuffle, _ = get_no_augmentation(
                            self.dl_tr_unlabeled_false_shuffle, self.dl_tr_unlabeled_false_shuffle,
                            params=self.data_aug_params,
                            pin_memory=self.pin_memory,
                        )
                        with torch.no_grad():
                            for i in range(0, self.num_unlabeled_data, self.batch_size):
                                unlabeled_data_dict_false_shuffle = next(self.tr_gen_unlabeled_false_shuffle)
                                unlabeled_data_false_shuffle = unlabeled_data_dict_false_shuffle['data']
                                unlabeled_data_false_shuffle = maybe_to_torch(unlabeled_data_false_shuffle)
                                if torch.cuda.is_available():
                                    unlabeled_data_false_shuffle = to_cuda(unlabeled_data_false_shuffle)
                                outputs_false_shuffle = self.network(unlabeled_data_false_shuffle)
                                outputs_false_shuffle1 = self.network1(unlabeled_data_false_shuffle)
                                outputs_false_shuffle = F.normalize(outputs_false_shuffle, p=2, dim=1)
                                outputs_false_shuffle1 = F.normalize(outputs_false_shuffle1, p=2, dim=1)

                                if i + self.batch_size < self.num_unlabeled_data:
                                    self.targets_ensemble[i : i + self.batch_size] = self.alpha * self.targets_ensemble[i : i + self.batch_size] + (1. - self.alpha) * outputs_false_shuffle.detach()
                                    self.targets_pseudo[i : i + self.batch_size] = self.targets_ensemble[i : i + self.batch_size] / (1. - self.alpha ** (self.epoch - 3 + 1))

                                    self.targets_ensemble1[i : i + self.batch_size] = self.alpha * self.targets_ensemble1[i : i + self.batch_size] + (1. - self.alpha) * outputs_false_shuffle1.detach()
                                    self.targets_pseudo1[i : i + self.batch_size] = self.targets_ensemble1[i : i + self.batch_size] / (1. - self.alpha ** (self.epoch - 3 + 1))
                                else:
                                    rest = int(self.num_unlabeled_data % self.batch_size)
                                    self.targets_ensemble[i : i + rest] = self.alpha * self.targets_ensemble[i : i + rest] + (1. - self.alpha) * outputs_false_shuffle[: rest].detach()
                                    self.targets_pseudo[i : i + rest] = self.targets_ensemble[i : i + rest] / (1. - self.alpha ** (self.epoch - 3 + 1))

                                    self.targets_ensemble1[i : i + rest] = self.alpha * self.targets_ensemble1[i : i + rest] + (1. - self.alpha) * outputs_false_shuffle1[: rest].detach()
                                    self.targets_pseudo1[i : i + rest] = self.targets_ensemble1[i : i + rest] / (1. - self.alpha ** (self.epoch - 3 + 1))
                            self.update_unlabeled_pseudo_labels = False
                            del self.tr_gen_unlabeled_false_shuffle
                else:
                    te_loss = torch.tensor(0.0)
                    te_loss1 = torch.tensor(0.0)
                
                l = sup_loss + consistency_loss + te_loss
                l1 = sup_loss1 + consistency_loss1 + te_loss1

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()

                self.amp_grad_scaler1.scale(l1).backward()
                self.amp_grad_scaler1.unscale_(self.optimizer1)
                torch.nn.utils.clip_grad_norm_(self.network1.parameters(), 12)
                self.amp_grad_scaler1.step(self.optimizer1)
                self.amp_grad_scaler1.update()
        else:
            # sup loss
            outputs = self.network(inputs)
            sup_loss = self.loss(outputs[: self.batch_size], target1)
            outputs1 = self.network1(inputs)
            sup_loss1 = self.loss(outputs1[: self.batch_size], target1)
            
            # pseudo loss
            consistency_loss = torch.tensor(0.0)
            consistency_loss1 = torch.tensor(0.0)
            if self.apply_consistency:
                pseudo_logits, pseudo_label = torch.max(softmax_helper(outputs[self.batch_size:].detach()), dim=1)
                pseudo_label = pseudo_label.unsqueeze(1).long()
                pseudo_logits1, pseudo_label1 = torch.max(softmax_helper(outputs1[self.batch_size:].detach()), dim=1)
                pseudo_label1 = pseudo_label1.unsqueeze(1).long()

                consistency_loss = consistency_weight * self.loss(outputs[self.batch_size:], pseudo_label1.detach())
                consistency_loss1 = consistency_weight * self.loss(outputs1[self.batch_size:], pseudo_label.detach())
            else:
                consistency_loss = torch.tensor(0.0)
                consistency_loss1 = torch.tensor(0.0)

            # TE
            te_loss = torch.tensor(0.0)
            te_loss1 = torch.tensor(0.0)
            if self.apply_TE:
                outputs_ul_current = outputs[self.batch_size:]
                outputs1_ul_current = outputs1[self.batch_size:]
                
                if self.epoch > 2:
                    temp_ensemble = self.targets_pseudo[unlabeled_index].clone()
                    temp_ensemble = self.alpha * temp_ensemble + (1 - self.alpha) * F.normalize(outputs_ul_current.detach(), p=2, dim=1)
                    te_loss = consistency_weight * self.consistency_loss(outputs_ul_current, temp_ensemble)

                    temp_ensemble1 = self.targets_pseudo1[unlabeled_index].clone()
                    temp_ensemble1 = self.alpha * temp_ensemble1 + (1 - self.alpha) * F.normalize(outputs1_ul_current.detach(), p=2, dim=1)
                    te_loss1 = consistency_weight * self.consistency_loss(outputs1_ul_current, temp_ensemble1)

                if do_backprop is False and self.epoch > 1 and self.update_unlabeled_pseudo_labels and self.iter_step % self.num_batches_per_epoch == 0:   
                    self.dl_tr_unlabeled_false_shuffle.reset_pointers()
                    self.tr_gen_unlabeled_false_shuffle, _ = get_no_augmentation(
                        self.dl_tr_unlabeled_false_shuffle, self.dl_tr_unlabeled_false_shuffle,
                        params=self.data_aug_params,
                        pin_memory=self.pin_memory,
                    )
                    with torch.no_grad():
                        for i in range(0, self.num_unlabeled_data, self.batch_size):
                            unlabeled_data_dict_false_shuffle = next(self.tr_gen_unlabeled_false_shuffle)
                            unlabeled_data_false_shuffle = unlabeled_data_dict_false_shuffle['data']
                            unlabeled_data_false_shuffle = maybe_to_torch(unlabeled_data_false_shuffle)
                            if torch.cuda.is_available():
                                unlabeled_data_false_shuffle = to_cuda(unlabeled_data_false_shuffle)
                            outputs_false_shuffle = self.network(unlabeled_data_false_shuffle)
                            outputs_false_shuffle1 = self.network1(unlabeled_data_false_shuffle)
                            outputs_false_shuffle = F.normalize(outputs_false_shuffle, p=2, dim=1)
                            outputs_false_shuffle1 = F.normalize(outputs_false_shuffle1, p=2, dim=1)

                            if i + self.batch_size < self.num_unlabeled_data:
                                self.targets_ensemble[i : i + self.batch_size] = self.alpha * self.targets_ensemble[i : i + self.batch_size] + (1. - self.alpha) * outputs_false_shuffle.detach()
                                self.targets_pseudo[i : i + self.batch_size] = self.targets_ensemble[i : i + self.batch_size] / (1. - self.alpha ** (self.epoch - 2 + 1))

                                self.targets_ensemble1[i : i + self.batch_size] = self.alpha * self.targets_ensemble1[i : i + self.batch_size] + (1. - self.alpha) * outputs_false_shuffle1.detach()
                                self.targets_pseudo1[i : i + self.batch_size] = self.targets_ensemble1[i : i + self.batch_size] / (1. - self.alpha ** (self.epoch - 2 + 1))
                            else:
                                rest = int(self.num_unlabeled_data % self.batch_size)
                                self.targets_ensemble[i : i + rest] = self.alpha * self.targets_ensemble[i : i + rest] + (1. - self.alpha) * outputs_false_shuffle[: rest].detach()
                                self.targets_pseudo[i : i + rest] = self.targets_ensemble[i : i + rest] / (1. - self.alpha ** (self.epoch - 2 + 1))

                                self.targets_ensemble1[i : i + rest] = self.alpha * self.targets_ensemble1[i : i + rest] + (1. - self.alpha) * outputs_false_shuffle1[: rest].detach()
                                self.targets_pseudo1[i : i + rest] = self.targets_ensemble1[i : i + rest] / (1. - self.alpha ** (self.epoch - 2 + 1))
                        self.update_unlabeled_pseudo_labels = False
                        del self.tr_gen_unlabeled_false_shuffle
            else:
                te_loss = torch.tensor(0.0)
                te_loss1 = torch.tensor(0.0)
            
            l = sup_loss + consistency_loss + te_loss
            l1 = sup_loss1 + consistency_loss1 + te_loss1

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

                l1.backward()
                torch.nn.utils.clip_grad_norm_(self.network1.parameters(), 12)
                self.optimizer1.step()

        if run_online_evaluation:
            self.run_online_evaluation_1(outputs[: self.batch_size], target1)
            self.run_online_evaluation_2(outputs1[: self.batch_size], target1)

        if do_backprop:
            self.iter_step = self.iter_step + 1

        return l.detach().cpu().numpy(), sup_loss.detach().cpu().numpy(), consistency_loss.detach().cpu().numpy(), te_loss.detach().cpu().numpy(), \
            l1.detach().cpu().numpy(), sup_loss1.detach().cpu().numpy(), consistency_loss1.detach().cpu().numpy(), te_loss1.detach().cpu().numpy()

    def do_split(self):
        """
        Copy from nnUNetTrainerV2 but provide the number of training cases

        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
            self.print_to_log_file("Using all training data during training..., len(tr_keys)=", str(len(tr_keys)))
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        assert isinstance(self.num_training_cases, (str, int)), "num_training_cases must be int or 'all'"

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        self.dataset_tr_unlabeled = OrderedDict()
        if self.num_training_cases == 'all':
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset[i]
        else:
            for i in tr_keys[:self.num_training_cases]:
                self.dataset_tr[i] = self.dataset[i]
            for i in tr_keys[self.num_training_cases:]: # the rest of training data is treated as unlabeled data
                self.dataset_tr_unlabeled[i] = self.dataset[i]
            self.print_to_log_file('select %d cases for training' % (len(self.dataset_tr.keys())))
            self.print_to_log_file('training cases: {}\n'.format(self.dataset_tr.keys()))
            self.print_to_log_file('select %d cases for unlabeled data' % (len(self.dataset_tr_unlabeled.keys())))
        
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        Copy from nnUNetTrainerV2 but set self.deep_supervision_scales to None

        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = None

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

        self.data_aug_params["to_tensor"] = True
        self.data_aug_params["contain_seg"] = True

        self.basic_generator_patch_size = self.patch_size

    def on_epoch_end(self):
        """
        Copy from nnUNetTrainerV2, discard the limitation of max_num_epochs

        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        ret = super(nnUNetTrainerV2, self).on_epoch_end()

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return ret

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="total_loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
            
            self.summary_writer.add_scalar("total_tr_loss", self.all_tr_losses[-1], self.epoch)
            self.summary_writer.add_scalar("tr_sup_loss", self.all_tr_sup_losses[-1], self.epoch)
            self.summary_writer.add_scalar("tr_consistency_loss", self.all_tr_consistency_losses[-1], self.epoch)
            self.summary_writer.add_scalar("tr_te_loss", self.all_tr_te_losses[-1], self.epoch)

            self.summary_writer.add_scalar("total_val_loss", self.all_val_losses[-1], self.epoch)
            self.summary_writer.add_scalar("val_sup_loss", self.all_val_sup_losses[-1], self.epoch)
            self.summary_writer.add_scalar("val_consistency_loss", self.all_val_consistency_losses[-1], self.epoch)
            self.summary_writer.add_scalar("val_te_loss", self.all_val_te_losses[-1], self.epoch)

            self.summary_writer.add_scalar("total_tr_loss1", self.all_tr_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("tr_sup_loss1", self.all_tr_sup_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("tr_consistency_loss1", self.all_tr_consistency_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("tr_te_loss1", self.all_tr_te_losses1[-1], self.epoch)

            self.summary_writer.add_scalar("total_val_loss1", self.all_val_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("val_sup_loss1", self.all_val_sup_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("val_consistency_loss1", self.all_val_consistency_losses1[-1], self.epoch)
            self.summary_writer.add_scalar("val_te_loss1", self.all_val_te_losses1[-1], self.epoch)

            self.summary_writer.add_scalar("lr", np.round(self.optimizer.param_groups[0]['lr'], decimals=6), self.epoch)
            self.summary_writer.add_scalar("lr1", np.round(self.optimizer1.param_groups[0]['lr'], decimals=6), self.epoch)

            if len(self.all_val_eval_metrics) == len(x_values):
                self.summary_writer.add_scalar("evaluation metric", self.all_val_eval_metrics[-1], self.epoch)
            
            if len(self.all_val_eval_metrics1) == len(x_values):
                self.summary_writer.add_scalar("evaluation metric1", self.all_val_eval_metrics1[-1], self.epoch)
            
            if self.epoch == self.max_num_epochs - 1:
                self.summary_writer.close()

        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()
            self.amp_grad_scaler1 = GradScaler()

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        self.optimizer1.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr1:", np.round(self.optimizer1.param_groups[0]['lr'], decimals=6))

    def run_training(self):
        """
        This is a copy from nnUNetTrainerV2 but set the self.network.do_ds to False

        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()
        _ = self.tr_gen_unlabeled.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)
        
        self.iter_step = 0
        self.best_val = 9999
        self.best_val1 = 9999
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            train_sup_losses_epoch = []
            train_consistency_losses_epoch = []
            train_te_losses_epoch = []

            train_losses1_epoch = []
            train_sup_losses1_epoch = []
            train_consistency_losses1_epoch = []
            train_te_losses1_epoch = []

            # train one epoch
            self.network.train()
            self.network1.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l, sup_l, consistency_l, te_l, l1, sup_l1, consistency_l1, te_l1 = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                        train_sup_losses_epoch.append(sup_l)
                        train_consistency_losses_epoch.append(consistency_l)
                        train_te_losses_epoch.append(te_l)

                        train_losses1_epoch.append(l1)
                        train_sup_losses1_epoch.append(sup_l1)
                        train_consistency_losses1_epoch.append(consistency_l1)
                        train_te_losses1_epoch.append(te_l1)

            else:
                for _ in range(self.num_batches_per_epoch):
                    l, sup_l, consistency_l, te_l, l1, sup_l1, consistency_l1, te_l1 = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)
                    train_sup_losses_epoch.append(sup_l)
                    train_consistency_losses_epoch.append(consistency_l)
                    train_te_losses_epoch.append(te_l)

                    train_losses1_epoch.append(l1)
                    train_sup_losses1_epoch.append(sup_l1)
                    train_consistency_losses1_epoch.append(consistency_l1)
                    train_te_losses1_epoch.append(te_l1)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.all_tr_sup_losses.append(np.mean(train_sup_losses_epoch))
            self.all_tr_consistency_losses.append(np.mean(train_consistency_losses_epoch))
            self.all_tr_te_losses.append(np.mean(train_te_losses_epoch))

            self.all_tr_losses1.append(np.mean(train_losses1_epoch))
            self.all_tr_sup_losses1.append(np.mean(train_sup_losses1_epoch))
            self.all_tr_consistency_losses1.append(np.mean(train_consistency_losses1_epoch))
            self.all_tr_te_losses1.append(np.mean(train_te_losses1_epoch))
            self.print_to_log_file("train loss : %.4f, train sup loss : %.4f, train consistency loss : %.4f, train te loss : %.4f; train loss1 : %.4f, train sup loss1 : %.4f, train consistency loss1 : %.4f, train te loss1 : %.4f" % (
                self.all_tr_losses[-1], self.all_tr_sup_losses[-1], self.all_tr_consistency_losses[-1], self.all_tr_te_losses[-1], \
                    self.all_tr_losses1[-1], self.all_tr_sup_losses1[-1], self.all_tr_consistency_losses1[-1], self.all_tr_te_losses1[-1]))

            if self.validate_in_training:
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    self.network1.eval()
                    val_losses = []
                    val_sup_losses = []
                    val_consistency_losses = []
                    val_te_losses = []
                    val_losses1 = []
                    val_sup_losses1 = []
                    val_consistency_losses1 = []
                    val_te_losses1 = []
                    self.update_unlabeled_pseudo_labels = True
                    for b in range(self.num_val_batches_per_epoch):
                        l, sup_l, consistency_l, te_l, l1, sup_l1, consistency_l1, te_l1 = self.run_iteration(self.val_gen, False, True)
                        val_losses.append(l)
                        val_sup_losses.append(sup_l)
                        val_consistency_losses.append(consistency_l)
                        val_te_losses.append(te_l)

                        val_losses1.append(l1)
                        val_sup_losses1.append(sup_l1)
                        val_consistency_losses1.append(consistency_l1)
                        val_te_losses1.append(te_l1)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.all_val_sup_losses.append(np.mean(val_sup_losses))
                    self.all_val_consistency_losses.append(np.mean(val_consistency_losses))
                    self.all_val_te_losses.append(np.mean(val_te_losses))
                    self.all_val_losses1.append(np.mean(val_losses1))
                    self.all_val_sup_losses1.append(np.mean(val_sup_losses1))
                    self.all_val_consistency_losses1.append(np.mean(val_consistency_losses1))
                    self.all_val_te_losses1.append(np.mean(val_te_losses1))
                    self.print_to_log_file("val loss : %.4f, val sup loss : %.4f, val consistency loss : %.4f, val te loss : %.4f; val loss1 : %.4f, val sup loss1 : %.4f, val consistency loss1 : %.4f, val te loss1 : %.4f" % (
                        self.all_val_losses[-1], self.all_val_sup_losses[-1], self.all_val_consistency_losses[-1], self.all_val_te_losses[-1], \
                            self.all_val_losses1[-1], self.all_val_sup_losses1[-1], self.all_val_consistency_losses1[-1], self.all_val_te_losses1[-1]))

                    # save the best checkpoint according to validation set
                    if self.all_val_losses[-1] < self.best_val:
                        self.best_val = self.all_val_losses[-1]
                        self.save_checkpoint(join(self.output_folder, "model_best_checkpoint_1.model"))
                    if self.all_val_losses1[-1] < self.best_val1:
                        self.best_val1 = self.all_val_losses1[-1]
                        self.save_checkpoint(join(self.output_folder, "model_best_checkpoint_2.model"))

                    if self.also_val_in_tr_mode:
                        self.network.train()
                        # validation with train=True
                        val_losses = []
                        for b in range(self.num_val_batches_per_epoch):
                            l, sup_l, consistency_l = self.run_iteration(self.val_gen, False)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        
        self.print_to_log_file("best_val: %.4f, best_val1: %.4f" % (self.best_val, self.best_val1))

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        if self.network1 is not None:
            state_dict1 = self.network1.state_dict()
            for key in state_dict1.keys():
                state_dict1[key] = state_dict1[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
            optimizer_state_dict1 = self.optimizer1.state_dict()
        else:
            optimizer_state_dict = None
            optimizer_state_dict1 = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_tr_sup_losses, self.all_tr_consistency_losses, self.all_tr_te_losses, \
                self.all_tr_losses1, self.all_tr_sup_losses1, self.all_tr_consistency_losses1, self.all_tr_te_losses1, \
                    self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()
        if self.network1 is not None:
            save_this['state_dict1'] = state_dict1
            save_this['optimizer_state_dict1'] = optimizer_state_dict1

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))
        
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        # network1
        new_state_dict1 = OrderedDict()
        curr_state_dict_keys1 = list(self.network1.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict1'].items():
            key = k
            if key not in curr_state_dict_keys1 and key.startswith('module.'):
                key = key[7:]
            new_state_dict1[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.network1.load_state_dict(new_state_dict1)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            optimizer_state_dict1 = checkpoint['optimizer_state_dict1']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if optimizer_state_dict1 is not None:
                self.optimizer1.load_state_dict(optimizer_state_dict1)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_tr_sup_losses, self.all_tr_consistency_losses, self.all_tr_te_losses, \
                self.all_tr_losses1, self.all_tr_sup_losses1, self.all_tr_consistency_losses1, self.all_tr_te_losses1, \
                    self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ret = super(nnUNetTrainerV2, self).validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        temp_network = self.network
        self.network = self.network1
        ret = super(nnUNetTrainerV2, self).validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name + '1', debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network = temp_network
        return ret


class nnUNetTrainerV2_SECT_4TrainCases(nnUNetTrainerV2_SECT):
    """
    custom_trainer_v1
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, num_training_cases=4):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, num_training_cases)
        self.alpha = 0.6
        self.max_num_epochs = 40
        self.consistency_rampup = 20.0

