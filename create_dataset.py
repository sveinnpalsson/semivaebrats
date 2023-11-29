import argparse
import os
from os.path import join
from random import shuffle

import numpy as np
import nibabel as nib
import pandas
import scipy.ndimage as aug
from tqdm import tqdm

def augment_image(image, max_angle):
    """
    Augments the image by rotating the image by max_angle in the axial plane in both directions
    Also flips the image from left to right and rotates by max_angle in both directions
    """
    angles = [-max_angle, max_angle]
    axes = [(0,1), (0,2), (1,2)]
    images_aug = [image,image[::-1]]
    for angle in angles:
        for axis in axes:
            images_aug.append(aug.rotate(image, angle, axes=axis, reshape=False, order=0))
            images_aug.append(aug.rotate(image[::-1], angle, axes=axis, reshape=False, order=0))
    return images_aug

                  
def create_data_subset(out_path, labels, files, augment=False, aug_angle=5, all_tumor_labels=True):
    cnt = 0
    labels_out = [] #will be different from input labels if using augmentation
    nf = len(files)
    labeled = len(labels)>0
    for i in tqdm(range(nf)):
        img = nib.load(files[i])
        if all_tumor_labels:
            data = np.array(img.get_fdata()).astype(float)
            data[np.where(data == 4)] = 3
        else:
            data = np.array(img.get_fdata() > 0).astype(float)

        if augment:
            images = augment_image(data,aug_angle)
        else:
            images = [data]

        for aug_num, image in enumerate(images):
            if labeled:
                labels_out.append(labels[i])
            subj_id = files[i].split("_seg.nii")[0].split('_')[-1]
            file_name = join(out_path, f'subj_{subj_id}_{aug_num}.nii.gz')
            nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), file_name)
            cnt += 1
    
    if labeled:
        np.save(join(out_path.replace("Labeled",""), 'labels'), np.array(labels_out))
    return cnt

def isnum(x):
    try:
        a=int(x)
        return True
    except:
        return False
    
def categorize(y):
    """
    Creates a survival class label list from a list of survival days
    categories are 0 for under 10 month survival, 1 for 10-15 months and 2 for 15+ months
    """
    y_out = []
    for yi in y:
        if int(yi)<(365*10.0)/12.0:
            y_out.append(0)
        elif int(yi)<(365*15.0)/12.0:
            y_out.append(1)
        else:
            y_out.append(2)
    return np.array(y_out)
                  
def categorize_resection(y):
    y_out = []
    for yi in y:
        if not type(yi) is str:
            y_out.append(0)
        elif yi == 'GTR':
            y_out.append(1)
        else:
            y_out.append(2)
    return np.array(y_out)

def create_dataset(args):
    np.random.seed(args.seed)
    data_dir = args.data_dir
    output_dir = args.output_dir

    fileformat = ".nii.gz"
    files = [join(data_dir, i, f"{i}_seg{fileformat}") for i in os.listdir(data_dir) if os.path.isdir(join(data_dir, i))]

    np.random.shuffle(files)
    file_ids = [i.split(os.sep)[-1].replace(".nii.gz","").replace("_seg","") for i in files]

    path_train_l = join(output_dir,"Train","Labeled")
    path_train_u = join(output_dir,"Train","Unlabeled")
    path_val = join(output_dir,"Validation","Labeled")
    
    if args.train_df == "":
        if os.path.isfile(join(data_dir,"survival_info.csv")):
             args.train_df = join(data_dir,"survival_info.csv")
        else:
             raise RuntimeError("dataframe not found")
    
    exist_ok = True
    os.makedirs(path_train_l,exist_ok=exist_ok)
    os.makedirs(path_train_u,exist_ok=exist_ok)
    os.makedirs(path_val,exist_ok=exist_ok)

    y = pandas.read_csv(args.train_df)
    if "brats_id" in y.columns.values:
        label_ids = y["brats_id"].values
        label_list = list(categorize(y["survival"].values))
    elif "Brats20ID" in y.columns.values:
        label_ids = y["Brats20ID"].values
        label_list = y["Survival_days"].values
        nums = [isnum(i) for i in label_list]
        label_ids = label_ids[nums]
        label_list = list(categorize(label_list[nums]))
    elif "BraTS21ID" in y.columns.values:
        label_ids = y["BraTS21ID"].values
        label_list = y["Survival_days"].values
        nums = [isnum(i) for i in label_list]
        label_ids = label_ids[nums]
        label_ids = ["BraTS2021_{:0>5}".format(i) for i in label_ids]
        label_list = list(categorize(label_list[nums]))
    else:
        raise RuntimeError("can't find brats-id column in dataframe")
    
    label_ids_list = list(label_ids)
    files_labeled = [files[i] for i in range(len(files)) if file_ids[i] in label_ids]
    files_unlabeled = [files[i] for i in range(len(files)) if file_ids[i] not in label_ids]
    n_u = len(files_unlabeled)
    n_l_train = int(len(files_labeled)*args.split)
    n_l_val = len(files_labeled) - n_l_train
    files_labeled_train = files_labeled[:n_l_train]
    files_labeled_val = files_labeled[n_l_train:]
    labels = [label_list[label_ids_list.index(i)] for i in file_ids if i in label_ids]
    labels_train = labels[:n_l_train]
    labels_val = labels[n_l_train:]

    print("total files: %d, labeled: %d, unlabeled: %d"%(len(files),len(files_labeled),n_u))
    print("train labeled: %d, validation labeled: %d"%(n_l_train,n_l_val))

    print("label distribution train: ",np.bincount(labels_train)/len(labels_train))
    print("label distribution val: ",np.bincount(labels_val)/len(labels_val))

    cont = input("Continue (y) / Abort (n): ")
    if not cont.lower()=="y":
        print("aborted")
        return

    val_cnt = create_data_subset(path_val, labels_val, files_labeled_val,
                       augment=args.augment_val, aug_angle=args.angle, all_tumor_labels=args.all_tumor_labels)
    print("Saved %d validation images"%val_cnt)

    l_cnt = create_data_subset(path_train_l, labels_train, files_labeled_train,
                               augment=args.augmentation, aug_angle=args.angle, all_tumor_labels=args.all_tumor_labels)
    print("Saved %d labeled training images"%l_cnt)

    
    u_cnt = create_data_subset(path_train_u, [], files_unlabeled,
                               augment=args.augmentation,aug_angle=args.angle,all_tumor_labels=args.all_tumor_labels)
    print("Saved %d labeled training images"%u_cnt)

    open(join(args.output_dir,"seed.txt"),"w").write("seed=%d"%args.seed)

    y.to_csv(join(args.output_dir, "info_table.csv"), index=False)



def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='MICCAI_BraTS2020_TrainingData', metavar='DATA_DIR',
                        help="path to BraTS data folder")
    parser.add_argument('--output_dir', type=str, default='data/brats_20_semivae_dataset/', metavar='OUTPUT_DIR',
                        help="output directory")
    parser.add_argument('--train_df', type=str, default='', metavar='OUTPUT_DIR',
                        help="BraTS dataframe")
    parser.add_argument('--split', type=float, default='0.8', metavar='SPLIT',
                        help="proportion of train images. (1-split) proportion of validation images")
    parser.add_argument('--augmentation', type=bool, default=False, metavar='AUG',
                        help="Augmentation (rotate and flip) ")
    parser.add_argument('--augment_val', type=bool, default=False, metavar='AUGVAL',
                        help="Augment validation set")
    parser.add_argument('--angle', type=float, default=10, metavar='ANGLE',
                        help="angle of rotation for augmentation")
    parser.add_argument('--all_tumor_labels', type=bool, default=True, metavar='categorical',
                        help="Use all the different tumor structures")
    parser.add_argument('--seed', type=int, default='1337', metavar='ANGLE',
                        help="angle of rotation for augmentation")
    args = parser.parse_args()
    
    create_dataset(args)


if __name__ == '__main__':
    main()
