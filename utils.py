from datetime import datetime as dt
import math
import os
import random
from os.path import join
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
import torch
import torch.nn.functional as f
from torch import nn
from tqdm import tqdm

def set_rnd_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # maybe we don't need this one
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def match_args(a, b):
    mismatch_ = False
    if not a.batch_size == b.batch_size:
        mismatch_ = True
    if not a.data_dir_train == b.data_dir_train:
        mismatch_ = True
    if not a.zdim == b.zdim:
        mismatch_ = True
    if not a.nf == b.nf:
        mismatch_ = True
    return not mismatch_

def find_recent_matching_args(args):
    print("recent runs with matching input arguments")
    for f_ in os.listdir(args.log_dir):
        try:
            f_time = dt.strptime(f_, "%Y-%m-%d-%H-%M-%S")
        except:
            continue
        if (dt.now()-f_time).days < 5:
            try:
                if os.path.isfile(join(args.log_dir, f_, "args.pkl")):
                    with open(join(args.log_dir, f_, "args.pkl"), "rb") as argsf:
                        argsf = pickle.load(argsf)
                    if match_args(argsf, args):
                        print(f_)
            except:
                continue

def check_args(args):
    if args.load == "":
        tmppath = join(args.log_dir, 'args.pkl')
        with open(tmppath, 'wb') as file:
            pickle.dump(args, file)
    else:
        with open(join(args.log_dir, "args.pkl"), "rb") as file:
            previous_args = pickle.load(file)
        mismatch = False
        if not previous_args.data_dir_train == args.data_dir_train:
            print("[!] data directory (%s) not same as in previous run (%s)" % (args.data_dir_train, previous_args.data_dir_train))
            mismatch = True
        if not previous_args.zdim == args.zdim:
            print("[!] z dimension (%d) not same as in previous run (%d)" % (args.zdim, previous_args.zdim))
            mismatch = True
        if not previous_args.nf == args.nf:
            print("[!] model size (nf=%d) not same as in previous run (nf=%d)" % (args.nf, previous_args.nf))
            mismatch = True
        if mismatch:
            find_recent_matching_args(args)
            raise RuntimeError


def one_hot(y, c):
    y_out = np.zeros((len(y), c + 1))
    for i in range(len(y)):
        y_out[i, y[i]] = 1
    return np.array(y_out).astype(np.int8)


def one_hot_data(y, c):
    targets = np.array(y.reshape(-1), np.uint8)
    one_hot_targets = np.eye(c+1, dtype=np.uint8)[targets]
    return one_hot_targets.reshape([y.shape[0], y.shape[1], y.shape[2], y.shape[3], c + 1])


def print_num_params(model, show_norm=False):
    print("\n--- Trainable parameters:")
    num_params_tot = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            num_params_tot += num_params
            if show_norm:
                norm = "{:.3g}".format((param.detach() ** 2).sum().sqrt().item() / num_params)
            else:
                norm = ''
            print("{:6d}  {:43}  {}".format(num_params, name, norm))
    print("  - Total trainable parameters:", num_params_tot)
    print("---------\n")


def pad_img(x, target_size):
    paddings = []
    for size in reversed(x.shape[2:]):
        pad1 = (target_size - size) // 2
        pad2 = target_size - size - pad1
        paddings.extend([pad1, pad2])
    return f.pad(x, paddings)


def from_np(*inputs, device=None):
    def from_np_(x, device_):
        x = torch.from_numpy(x)
        if device_ is not None:
            x = x.to(device_)
        return x

    assert isinstance(inputs, tuple)  # just to make sure
    out = tuple(from_np_(x, device) for x in inputs)
    if len(out) == 1:
        out = out[0]
    return out

def load_dataset_subfolder(data_dir_train, data_shape, crop=True):
    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    width, height, depth = data_shape
    files_train = list(files(data_dir_train))
    files_train.sort()  # sort alphabetically
    files_train.sort(key=len)  # and then by length so that labels and images are the same
    number_files = len(files_train)
    x_data = np.empty([number_files, width, height, depth], np.uint8)
    x_ids = []
    for i in tqdm(range(number_files)):
        image_path = join(data_dir_train, files_train[i])
        x_data[i] = nib.load(image_path).get_fdata()[0:width, 0:height, 0:depth]
        x_ids.append(image_path.split('subj_')[1].split('_')[0])

    x_data = x_data.reshape([-1, width, height, depth, 1])
    if crop:
        nonzero = np.count_nonzero(x_data)
        x_data = x_data[:, 49:-45, 29:-23, 15:-11]
        print("excluded %d/%d nonzero values when cropping" % (nonzero-np.count_nonzero(x_data), nonzero))

    return x_data, x_ids


def load_and_preproc_data(data_dir_train, data_dir_val, data_shape,
                 crop=True, binary_input=False, process_in_batches=False):

    try:
        print("Attemping to load preprocessed dataset")
        dataset_dict = np.load(join(data_dir_train.replace("/Train", ""), "preprocessed_dataset.npz"))
        x_l = dataset_dict['x_l']
        x_u = dataset_dict['x_u']
        y_l = dataset_dict['y_l']
        x_v = dataset_dict['x_v']
        y_v = dataset_dict['y_v']
        y_dim = dataset_dict['y_dim']
        x_l_ids = dataset_dict['x_l_ids']
        x_u_ids = dataset_dict['x_u_ids']
        x_v_ids = dataset_dict['x_v_ids']
    except FileNotFoundError:
        print("Failed: loading from scratch")
        print("   -> unlabeled training set")
        x_u, x_u_ids = load_dataset_subfolder(join(data_dir_train, 'Unlabeled'), data_shape, crop=crop)
        print("   -> labeled training set")
        x_l, x_l_ids = load_dataset_subfolder(join(data_dir_train, 'Labeled'), data_shape, crop=crop)
        y_l = np.load(join(data_dir_train, 'labels.npy'))

        print("   -> validation set")
        x_v, x_v_ids = load_dataset_subfolder(join(data_dir_val, 'Labeled'), data_shape, crop=crop)
        y_v = np.load(join(data_dir_val, 'labels.npy'))

        y_dim = np.max(y_l) + 1
        y_l = one_hot(y_l, y_dim - 1)
        y_v = one_hot(y_v, y_dim - 1)

        if binary_input:
            print("   -> make data in one hot format")
            x_u = one_hot_data(x_u, x_u.max())
            x_l = one_hot_data(x_l, x_l.max())
            x_v = one_hot_data(x_v, x_v.max())

        print("Dataset loaded")

        if binary_input:
            reduce_func = np.mean
        else:
            reduce_func = np.median

        if process_in_batches:   # for memory problems
            batch_size = 200
            print("   processing unlabeled data in batches of", batch_size)

            # Save current data in temp files
            ds_size = len(x_u)
            print("      saving temp data batches")
            for i in range(0, ds_size, batch_size):
                fname = join(data_dir_train.replace("/Train", ""), 'tmp_' + str(i))
                np.savez_compressed(fname, data=x_u[i:i+batch_size])

            # Pad (1, 1, 0) and then reduce by 2x2x2
            x_u = []
            print("      processing temp data batches")
            for i in tqdm(range(0, ds_size, batch_size)):
                fname = join(data_dir_train.replace("/Train", ""), 'tmp_' + str(i) + '.npz')
                tmp_data = np.load(fname)['data']
                tmp_data = reduce_data(tmp_data, reduce_func)
                x_u.append(tmp_data)
            x_u = np.concatenate(x_u, axis=0)

        else:
            x_u = reduce_data(x_u, reduce_func)

        x_l = reduce_data(x_l, reduce_func)
        x_v = reduce_data(x_v, reduce_func)
        print(x_u.shape)
        print("Downsampling completed")

        np.savez_compressed(join(data_dir_train.replace("/Train", ""), 'preprocessed_dataset'),
                            x_l=x_l, x_u=x_u, y_l=y_l, x_v=x_v, y_v=y_v, y_dim=y_dim, x_l_ids=x_l_ids,
                            x_u_ids=x_u_ids, x_v_ids=x_v_ids)
        print("Dataset saved")

    x_l = np.transpose(x_l, (0, 4, 1, 2, 3))
    x_u = np.transpose(x_u, (0, 4, 1, 2, 3))
    x_v = np.transpose(x_v, (0, 4, 1, 2, 3))

    data = {}
    data['x_l'], data['x_u'], data['x_v'], data['y_l'], data['y_v'], data['y_dim'] = x_l, x_u, x_v, y_l, y_v, y_dim
    data['x_l_ids'], data['x_u_ids'], data['x_v_ids'] = x_l_ids, x_u_ids, x_v_ids
    return data

    
def combine_additional_data(age_l, age_u, age_v, rs_l, rs_u, rs_v, mgmt_l, mgmt_u, mgmt_v):
    # Initialize empty arrays
    c_l, c_u, c_v = np.array([]), np.array([]), np.array([])

    # Concatenate available data
    data_available = [data for data in [(age_l, age_u, age_v), (rs_l, rs_u, rs_v), (mgmt_l, mgmt_u, mgmt_v)] if data[0] is not None]
    if data_available:
        c_l = np.concatenate([data[0] for data in data_available], axis=1)
        c_u = np.concatenate([data[1] for data in data_available], axis=1)
        c_v = np.concatenate([data[2] for data in data_available], axis=1)

    # Determine the dimension of the clinical data
    c_dim = c_l.shape[1] if c_l.size > 0 else 0

    return c_l, c_u, c_v, c_dim

def get_data(args, orig_data_shape):
    data = load_and_preproc_data(args.data_dir_train, args.data_dir_val, orig_data_shape, binary_input=args.binary_input)

    # Determine number of labels based on binary input argument
    n_labels = data['x_l'].shape[1] if args.binary_input else len(np.bincount(data['x_l'][:10].astype(np.int8).flatten()))
    data['n_labels'] = n_labels

    # Process additional data based on arguments
    info_table = pd.read_csv(args.data_info_path)
    age_l, age_u, age_v = process_age_data(data, info_table) if args.use_age else (None, None, None)
    rs_l, rs_u, rs_v = process_rs_data(data, info_table) if args.use_rs else (None, None, None)
    mgmt_l, mgmt_u, mgmt_v = process_mgmt_data(data, info_table) if args.use_mgmt else (None, None, None)

    c_l, c_u, c_v, c_dim = combine_additional_data(age_l, age_u, age_v, rs_l, rs_u, rs_v, mgmt_l, mgmt_u, mgmt_v)

    data['c_l'], data['c_u'], data['c_v'] = c_l, c_u, c_v
    data['c_dim'] = c_dim

    return data

def process_age_data(data, info_table):
    age_mean = np.nanmean(info_table['Age'].values)
    age_std = np.nanstd(info_table['Age'].values)
    idcol = [i for i in info_table.columns.values if 'brats' in i.lower()][0]
    if '_' in str(info_table[idcol].values[0]):
        ids_table = info_table[idcol].apply(lambda x: str(x.split('_')[-1])).values
    else:
        ids_table = np.array([str(i) for i in info_table[idcol].values])

    age_l = np.expand_dims(np.array([info_table['Age'][ids_table==i].values[0] if i in ids_table else age_mean for i in data['x_l_ids']]), 1)
    age_u = np.expand_dims(np.array([info_table['Age'][ids_table==i].values[0] if i in ids_table else age_mean for i in data['x_u_ids']]), 1)
    age_v = np.expand_dims(np.array([info_table['Age'][ids_table==i].values[0] if i in ids_table else age_mean for i in data['x_v_ids']]), 1)
    return (age_l - age_mean) / age_std, (age_u - age_mean) / age_std, (age_v - age_mean) / age_std

def process_rs_data(data, info_table):
    idcol = [i for i in info_table.columns.values if 'brats' in i.lower()][0]
    if '_' in str(info_table[idcol].values[0]):
        ids_table = info_table[idcol].apply(lambda x: str(x.split('_')[-1])).values
    else:
        ids_table = np.array([str(i) for i in info_table[idcol].values])
    table_rs = np.array([int(i.replace('GTR','1').replace('STR','2')) if type(i) is str else 0 for i in info_table['Extent_of_Resection'].values])
    rs_l = np.expand_dims(np.array([table_rs[ids_table==i][0] if i in ids_table else 0 for i in data['x_l_ids']]), 1)
    rs_u = np.expand_dims(np.array([table_rs[ids_table==i][0] if i in ids_table else 0 for i in data['x_u_ids']]), 1)
    rs_v = np.expand_dims(np.array([table_rs[ids_table==i][0] if i in ids_table else 0 for i in data['x_v_ids']]), 1)
    return rs_l, rs_u, rs_v

def process_mgmt_data(data, info_table):
    idcol = [i for i in info_table.columns.values if 'brats' in i.lower()][0]
    if '_' in str(info_table[idcol].values[0]):
        ids_table = info_table[idcol].apply(lambda x: str(x.split('_')[-1])).values
    else:
        ids_table = np.array([str(i) for i in info_table[idcol].values])
    
    table_mgmt = np.array([int(i) if np.isfinite(i) else 2 for i in info_table['MGMT_value'].values])
    mgmt_l = np.expand_dims(np.array([table_mgmt[ids_table==i][0] if i in ids_table else 0 for i in data['x_l_ids']]), 1)
    mgmt_u = np.expand_dims(np.array([table_mgmt[ids_table==i][0] if i in ids_table else 0 for i in data['x_u_ids']]), 1)
    mgmt_v = np.expand_dims(np.array([table_mgmt[ids_table==i][0] if i in ids_table else 0 for i in data['x_v_ids']]), 1)
    return mgmt_l, mgmt_u, mgmt_v


def to_rgb(im):
    if len(im.shape)==4:
        im=im[0]
    if len(im.shape)==2:
        new_shape = [3]+list(im.shape)
    else:
        new_shape = list(im.shape)
        new_shape[0] = 3
    rgb_im = torch.zeros(new_shape)
    rgb_im[0] = im==1
    rgb_im[1] = im==2
    rgb_im[2] = im==3
    return rgb_im

def reduce_data(data, reduce_func):
    data = np.pad(data, ((0, 0), (1, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
    data = block_reduce(data, (1, 2, 2, 2, 1), func=reduce_func)
    return data

def rotate_3d_batch(tensor, angle, axis):
    """
    Rotates a batch of 3D images by a specific angle around a specified axis.
    Assumes the input tensor is a PyTorch tensor with shape (N, C, D, H, W).
    
    :param tensor: Input tensor.
    :param angle: Rotation angle in degrees.
    :param axis: Axis to rotate around ('x', 'y', or 'z').
    :return: Rotated tensor.
    """
    # Create the 3D rotation matrix for the entire batch
    rot_matrix = get_3d_rotation_matrix(angle, axis, tensor.device)

    # Expand the rotation matrix to match the batch size
    rot_matrix = rot_matrix.expand(tensor.shape[0], 3, 3)

    # Add an extra column for translation (zeros)
    zeros = torch.zeros(tensor.shape[0], 3, 1, device=tensor.device)
    rot_matrix = torch.cat((rot_matrix, zeros), dim=2)

    # Apply the rotation to the entire batch
    N, C, D, H, W = tensor.size()
    grid = create_rotation_grid(rot_matrix, (N, C, D, H, W))
    rotated_tensor = torch.nn.functional.grid_sample(tensor, grid, mode='nearest', align_corners=True)
    
    return rotated_tensor

def get_3d_rotation_matrix(angle, axis, device):
    """
    Generates a 3D rotation matrix for the given angle and axis.
    """
    angle_rad = math.radians(angle)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)

    if axis == 'x':
        rot_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_val, -sin_val],
            [0, sin_val, cos_val]
        ], device=device)
    elif axis == 'y':
        rot_matrix = torch.tensor([
            [cos_val, 0, sin_val],
            [0, 1, 0],
            [-sin_val, 0, cos_val]
        ], device=device)
    elif axis == 'z':
        rot_matrix = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0],
            [0, 0, 1]
        ], device=device)
    else:
        raise ValueError("Invalid rotation axis")

    return rot_matrix

def create_rotation_grid(rot_matrix, size):
    """
    Create a grid for grid_sample from the rotation matrix.
    """
    N, C, D, H, W = size
    grid = torch.nn.functional.affine_grid(rot_matrix, torch.Size((N, C, D, H, W)), align_corners=True)
    return grid


### NN stuff

class Interpolate(nn.Module):
    def __init__(self, size=None, scale=None, mode='nearest'):
        super().__init__()
        assert (size is None and scale is not None or size is not None and scale is None)
        self.size = size
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return f.interpolate(x, size=self.size, scale_factor=self.scale, mode=self.mode)

class Crop3d(nn.Module):
    def __init__(self, *crop_amounts):
        super().__init__()
        self.crop_amounts = crop_amounts

    def forward(self, x):
        c = []
        for k in self.crop_amounts:
            c.append(k // 2)
            c.append(k - c[-1])
        return x[:, :, c[0]:x.size(2)-c[1], c[2]:x.size(3)-c[3], c[4]:x.size(4)-c[5]]

class Identity(nn.Module):
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        dims = (x.size(0), *self.dims)
        return x.view(*dims)

def print_shape(x):
    shp = tuple(x.shape)
    print("   current shape:", shp, " -  {} elements per sample".format(np.prod(shp) // shp[0]))
