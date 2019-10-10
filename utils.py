import os
import random
from os.path import join
import pickle
from datetime import datetime as dt

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as f
from skimage.measure import block_reduce
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

def load_dataset(data_dir_train, data_shape, return_names=False, crop=True):
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
    for i in tqdm(range(number_files)):
        image_path = join(data_dir_train, files_train[i])
        x_data[i] = nib.load(image_path).get_data()[0:width, 0:height, 0:depth]

    x_data = x_data.reshape([-1, width, height, depth, 1])
    if crop:
        nonzero = np.count_nonzero(x_data)
        x_data = x_data[:, 49:-45, 29:-23, 15:-11]
        print("excluded %d/%d nonzero values when cropping" % (nonzero-np.count_nonzero(x_data), nonzero))
    if return_names:
        return x_data, files_train
    return x_data


def get_all_data(data_dir_train, data_dir_val, data_shape,
                 crop=True, binary_input=False, process_in_batches=False):

    try:
        print("Attemping to load preprocessed dataset")
        dataset_dict = np.load(join(data_dir_train.replace("/Train", ""), "preprocessed_dataset.npz"))
        x_data_train_labeled = dataset_dict['x_data_train_labeled']
        x_data_train_unlabeled = dataset_dict['x_data_train_unlabeled']
        y_data_train_labeled = dataset_dict['y_data_train_labeled']
        x_data_val = dataset_dict['x_data_val']
        y_data_val = dataset_dict['y_data_val']
        y_dim = dataset_dict['y_dim']
    except FileNotFoundError:
        print("Failed: loading from scratch")
        print("   -> unlabeled training set")
        x_data_train_unlabeled = load_dataset(join(data_dir_train, 'Unlabeled'), data_shape, crop=crop)
        print("   -> labeled training set")
        x_data_train_labeled = load_dataset(join(data_dir_train, 'Labeled'), data_shape, crop=crop)
        y_data_train_labeled = np.load(join(data_dir_train, 'labels.npy'))

        print("   -> validation set")
        x_data_val = load_dataset(join(data_dir_val, 'Labeled'), data_shape, crop=crop)
        y_data_val = np.load(join(data_dir_val, 'labels.npy'))

        y_dim = np.max(y_data_train_labeled) + 1
        y_data_train_labeled = one_hot(y_data_train_labeled, y_dim - 1)
        y_data_val = one_hot(y_data_val, y_dim - 1)

        if binary_input:
            print("   -> make data in one hot format")
            x_data_train_unlabeled = one_hot_data(x_data_train_unlabeled, x_data_train_unlabeled.max())
            x_data_train_labeled = one_hot_data(x_data_train_labeled, x_data_train_labeled.max())
            x_data_val = one_hot_data(x_data_val, x_data_val.max())

        print("Dataset loaded")

        if binary_input:
            reduce_func = np.mean
        else:
            reduce_func = np.median

        if process_in_batches:   # for memory problems
            batch_size = 200
            print("   processing unlabeled data in batches of", batch_size)

            # Save current data in temp files
            ds_size = len(x_data_train_unlabeled)
            print("      saving temp data batches")
            for i in range(0, ds_size, batch_size):
                fname = join(data_dir_train.replace("/Train", ""), 'tmp_' + str(i))
                np.savez_compressed(fname, data=x_data_train_unlabeled[i:i+batch_size])

            # Pad (1, 1, 0) and then reduce by 2x2x2
            x_data_train_unlabeled = []
            print("      processing temp data batches")
            for i in tqdm(range(0, ds_size, batch_size)):
                fname = join(data_dir_train.replace("/Train", ""), 'tmp_' + str(i) + '.npz')
                tmp_data = np.load(fname)['data']
                tmp_data = reduce_data(tmp_data, reduce_func)
                x_data_train_unlabeled.append(tmp_data)
            x_data_train_unlabeled = np.concatenate(x_data_train_unlabeled, axis=0)

        else:
            x_data_train_unlabeled = reduce_data(x_data_train_unlabeled, reduce_func)

        x_data_train_labeled = reduce_data(x_data_train_labeled, reduce_func)
        x_data_val = reduce_data(x_data_val, reduce_func)
        print(x_data_train_unlabeled.shape)
        print("Downsampling completed")

        np.savez_compressed(join(data_dir_train.replace("/Train", ""), 'preprocessed_dataset'),
                            x_data_train_labeled=x_data_train_labeled,
                            x_data_train_unlabeled=x_data_train_unlabeled,
                            y_data_train_labeled=y_data_train_labeled,
                            x_data_val=x_data_val, y_data_val=y_data_val, y_dim=y_dim)
        print("Dataset saved")
    x_data_train_labeled = np.transpose(x_data_train_labeled, (0, 4, 1, 2, 3))
    x_data_train_unlabeled = np.transpose(x_data_train_unlabeled, (0, 4, 1, 2, 3))
    x_data_val = np.transpose(x_data_val, (0, 4, 1, 2, 3))
    return x_data_train_labeled, x_data_train_unlabeled, x_data_val, y_data_train_labeled, y_data_val, y_dim


def reduce_data(data, reduce_func):
    data = np.pad(data, ((0, 0), (1, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
    data = block_reduce(data, (1, 2, 2, 2, 1), func=reduce_func)
    return data


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

def print_shape(x, first_pass):
    if first_pass:
        shp = tuple(x.shape)
        print("   current shape:", shp, " -  {} elements per sample".format(np.prod(shp) // shp[0]))
