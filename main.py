import argparse
import os
import pickle
import sys
from datetime import datetime as dt
from os.path import join
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as f
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from utils import pad_img, print_num_params, from_np, get_all_data, one_hot, check_args

from model import SemiVAE


orig_data_shape = (239, 239, 154)   # width, height, depth
data_shape = (73, 94, 64)   # ~440k

grad_norm_clip = 200.0
log_interval = 100
alpha = 0.00001 * np.prod(data_shape)
gamma = 50

## Gumbel Softmax decay settings (exponential)
tau_start = 1.0
tau_end = 0.2
tau_steps = 50000

def main():
    # set_rnd_seed(31)   # reproducibility

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir_train', type=str, default='./data/brats_19/Train', metavar='DATA_TRAIN', help="data train directory")
    parser.add_argument('--data_dir_val', type=str, default='./data/brats_19/Validation', metavar='DATA_VAL', help="data validation directory")
    parser.add_argument('--log_dir', type=str, default='logs/', metavar='LOGS', help="logs directory")
    parser.add_argument('--models_dir', type=str, default='models/', metavar='MODELS',  help="models directory")
    parser.add_argument('--batch_size', type=int, default=16, metavar='BATCH', help="batch size")
    parser.add_argument('--learning_rate', type=float, default=2.0e-5, metavar='LR', help="learning rate")
    parser.add_argument('--epochs', type=int, default=1e6, metavar='EPOCHS', help="number of epochs")
    parser.add_argument('--zdim', type=int, default=16, metavar='ZDIM', help="Number of dimensions in latent space")
    parser.add_argument('--load', type=str, default='', metavar='LOADDIR', help="time string of previous run to load from")
    parser.add_argument('--binary_input', type=bool, default=False, metavar='BINARYINPUT', help="True=one input channel for each tumor structure")
    parser.add_argument('--use_age', type=bool, default=False, metavar='AGE', help="use age in prediction")
    parser.add_argument('--use_rs', type=bool, default=False, metavar='RESECTIONSTATUS', help="use resection status in prediction")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: {}".format(device))
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

    logdir_suffix = '-%s-zdim=%d-beta=5000-alpha=%.5f-lr=%.5f-gamma=%d-batch=%d'%(args.data_dir_train.replace("Train","").replace(".","").replace("/",""),args.zdim,alpha,args.learning_rate,gamma,args.batch_size)
    if args.use_age:
        logdir_suffix += "-age"
    if args.use_rs:
        logdir_suffix += "-rs"
    if args.binary_input:
        logdir_suffix += "-binary_input"
    if args.load == "":
        date_str = str(dt.now())[:-7].replace(":", "-").replace(" ", "-") + logdir_suffix
    else:
        date_str = args.load
    args.models_dir = join(args.models_dir, date_str)
    args.log_dir = join(args.log_dir, date_str)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    check_args(args)
    writer = SummaryWriter(args.log_dir + '-train')

    ## Get dataset

    data = get_all_data(args.data_dir_train, args.data_dir_val, orig_data_shape,binary_input=args.binary_input)

    x_data_train_labeled, x_data_train_unlabeled, x_data_val, y_data_train_labeled, y_data_val, y_dim = data
    if args.binary_input:
        n_labels = x_data_train_labeled.shape[1]
    else:
        n_labels = len(np.bincount(x_data_train_labeled[:10].astype(np.int8).flatten()))
    x_data_train_labeled = x_data_train_labeled.astype(np.int8)
    x_data_train_unlabeled = x_data_train_unlabeled.astype(np.int8)
    x_data_val = x_data_val.astype(np.int8)

    if args.use_age:
        age_std = 12.36
        age_mean = 62.2
        age_l = np.expand_dims(np.load(join(args.data_dir_train,"age_l.npy")),1)
        age_u = np.expand_dims(np.load(join(args.data_dir_train,"age_u.npy")),1)
        age_v = np.expand_dims(np.load(join(args.data_dir_val,"age.npy")),1)
        age_l = (age_l-age_mean)/age_std
        age_u = (age_u-age_mean)/age_std
        age_v = (age_v-age_mean)/age_std
    else:
        age_l, age_u, age_v = [],[],[]

    if args.use_rs:
        rs_l = one_hot(np.load(join(args.data_dir_train,"rs_l.npy")),2)
        rs_u = one_hot(np.load(join(args.data_dir_train,"rs_u.npy")),2)
        rs_v = one_hot(np.load(join(args.data_dir_val,"rs.npy")),2)
    else:
        rs_l, rs_u, rs_v = [],[],[]

    if args.use_rs and args.use_age:
        c_l = np.concatenate([age_l,rs_l],axis=1)
        c_u = np.concatenate([age_u,rs_u],axis=1)
        c_v = np.concatenate([age_v,rs_v],axis=1)
        c_dim = c_l.shape[1]
    elif args.use_rs:
        c_l,c_u,c_v = rs_l,rs_u,rs_v
        c_dim = c_l.shape[1]
    elif args.use_age:
        c_l,c_u,c_v = age_l,age_u,age_v
        c_dim = c_l.shape[1]
    else:
        c_l,c_u,c_v = np.array([]),np.array([]),np.array([])
        c_dim = 0

    y_data_val = y_data_val[:len(x_data_val)]
    print('x unlabeled data shape:', x_data_train_unlabeled.shape)
    print('x labeled data shape:', x_data_train_labeled.shape)
    print('x val data shape:', x_data_val.shape)
    assert data_shape == tuple(x_data_val.shape[2:])
    print('input labels: %d'%n_labels)

    model = SemiVAE(args.zdim, y_dim, c_dim,n_labels=n_labels,binary_input=args.binary_input).to(device)
    print_num_params(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0

    if args.load != "":
        print("Loading model from %s" % args.models_dir)
        nums = [int(i.split("_")[-1]) for i in os.listdir(args.models_dir)]
        start_epoch = max(nums)
        model_path = join(args.models_dir, "model_epoch_%d" % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'model_global_step' in checkpoint.keys():
            model.global_step = checkpoint['model_global_step']
        start_epoch = checkpoint['epoch']
        print("Loaded model at epoch %d, total steps: %d" % (start_epoch, model.global_step))

    t_start = dt.now()
    for epoch in range(int(start_epoch+1), int(args.epochs)):
        train(x_data_train_unlabeled, x_data_train_labeled, y_data_train_labeled,
              x_data_val, y_data_val, c_l, c_u, c_v, args.batch_size, epoch, model, optimizer, device, log_interval, writer, args.log_dir,n_labels)
        if (dt.now() - t_start).total_seconds() > 3600*2:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_global_step': model.global_step,
                    }, join(args.models_dir, "model_epoch_%d" % epoch))
            t_start = dt.now()
        sys.stdout.flush()   # need this when redirecting to file


def train(x_u, x_l, y_l, x_v, y_v, c_l, c_u, c_v, batch_size, epoch, model, optimizer, device, log_interv, writer, logdir,n_labels):
    model.train()
    loss_accum = []   # accumulate one epoch
    kl_accum = []
    classification_accum = []
    recons_accum = []
    h_accum = []
    accuracy_accum = []
    L_accum = []
    U_accum = []
    for train_step in range(len(x_u)//batch_size):
        optimizer.zero_grad()
        batch_idx_l = np.random.choice(len(x_l), batch_size, replace=False)
        batch_l = np.float32(x_l[batch_idx_l])
        batch_idx_u = np.random.choice(len(x_u), batch_size, replace=False)
        batch_u = np.float32(x_u[batch_idx_u])
        batch_labels = np.float32(y_l[batch_idx_l])
        
        if c_l.shape[0]!=0:
            batch_c_l = from_np(np.float32(c_l[batch_idx_l]),device=device)
            batch_c_u = from_np(np.float32(c_u[batch_idx_u]),device=device)
        else:
            batch_c_l = None
            batch_c_u = None
            
        batch_u, batch_l, batch_labels = from_np(batch_u, batch_l, batch_labels, device=device)
        
        ## Forward
        recon_batch_u, mu_u, logvar_u, logits_u = model(batch_u, [], batch_c_u, tau=tau_schedule(model.global_step))
        recon_batch_l, mu_l, logvar_l, logits_l = model(batch_l, batch_labels, batch_c_l, tau=tau_schedule(model.global_step))

        ## Losses (normalized by minibatch size)
        if n_labels<=2:
            bce_l = f.binary_cross_entropy(recon_batch_l, batch_l, reduction='sum') / batch_size
        else:
            if model.binary_input:
                bce_l = f.cross_entropy(recon_batch_l, torch.max(batch_l,1)[1].type(torch.int64), reduction='sum') / batch_size
            else:
                bce_l = f.cross_entropy(recon_batch_l, batch_l[:,0].type(torch.int64), reduction='sum') / batch_size

        kl_l = -0.5 * torch.sum(1 + logvar_l - mu_l.pow(2) - logvar_l.exp()) / batch_size
        loss_l = (bce_l + kl_l * beta_schedule(model.global_step)) / 2   # we're actually using 2 batches
        L_accum.append(loss_l.item())
        classification = f.cross_entropy(logits_l, torch.argmax(batch_labels, dim=1), reduction='sum') / batch_size
        loss_l += classification * alpha / 2    # in the overall loss it weighs half
        # TODO log p(y) is missing both here and in unlabeled (it's constant but we need it to report ELBO)

        accuracy = float(torch.sum(torch.max(logits_l, 1)[1].type(torch.cuda.FloatTensor) ==
                         torch.max(batch_labels, 1)[1].type(torch.cuda.FloatTensor))) / batch_size

        if n_labels<=2:
            bce_u = f.binary_cross_entropy(recon_batch_u, batch_u, reduction='sum') / batch_size
        else:
            if model.binary_input:
                bce_u = f.cross_entropy(recon_batch_u,torch.max(batch_u,1)[1].type(torch.int64), reduction='sum') / batch_size
            else:
                bce_u = f.cross_entropy(recon_batch_u, batch_u[:,0].type(torch.int64), reduction='sum') / batch_size

        kl_u = -0.5 * torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp()) / batch_size
        loss_u = (bce_u + kl_u * beta_schedule(model.global_step)) / 2   # we're actually using 2 batches
        softmax_u = torch.softmax(logits_u, dim=-1)
        h = -torch.sum(torch.mul(softmax_u, torch.log(softmax_u + 1e-12)), dim=-1).mean()
        loss_u += -h * gamma

        U_accum.append(loss_u.item())
        loss = loss_l + loss_u
        loss_accum.append(loss.item())
        kl_accum.append(kl_l.item() + kl_u.item())
        classification_accum.append(classification.item())
        accuracy_accum.append(accuracy)
        h_accum.append(h.item())
        recons_accum.append((bce_l.item() + bce_u.item()) / 2)

        ## Backward: accumulate gradients
        loss.backward()

        ## Clip gradients
        for param in model.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, grad_norm_clip)

        optimizer.step()
        model.global_step += 1

        ## Training step finished -- now write to tensorboard
        if model.global_step % log_interv == 0:
            ## Get losses over last step
            loss_step = loss_accum[-1]
            recons_step = np.mean(recons_accum[-1])
            kl_step = np.mean(kl_accum[-1])
            L_step = np.mean(L_accum[-1])
            U_step = np.mean(U_accum[-1])
            class_step = classification_accum[-1]
            accuracy_step = accuracy_accum[-1]
            h_step = h_accum[-1]
            print("epoch {}, step {} - loss: {:.4g} \trecons: {:.4g} \tKL: {:.4g} \tclass: {:.4g} \taccuracy: {:.4g} \tcategorical entropy: {:.4g}".format(
                    epoch, model.global_step, loss_step, recons_step, kl_step, class_step, accuracy_step, h_step))

            ## Save losses
            writer.add_scalar('losses/loss', loss_step, model.global_step)
            writer.add_scalar('losses/recons', recons_step, model.global_step)
            writer.add_scalar('losses/KL', kl_step, model.global_step)
            writer.add_scalar('losses/class', class_step, model.global_step)
            writer.add_scalar('losses/accuracy', accuracy_step, model.global_step)
            writer.add_scalar('losses/L', L_step, model.global_step)
            writer.add_scalar('losses/U', U_step, model.global_step)

        
        ## Validation set
        if model.global_step % (log_interv * 2) == 2:
            kl_accum_val = []
            classification_accum_val = []
            recons_accum_val = []
            accuracy_accum_val = []
            model.eval()
            for val_step in range(len(x_v)//batch_size):
                batch_idx = np.random.choice(len(x_v), batch_size, replace=False)
                data_val = np.float32(x_v[batch_idx])
                labels_val = np.float32(y_v[batch_idx])
                
                if c_v.shape[0]!=0:
                    c_val = from_np(np.float32(c_v[batch_idx]),device=device)
                else:
                    c_val = None
                data_val, labels_val = from_np(data_val, labels_val, device=device)
                recon_batch_val, mu_val, logvar_val, logits_val = model(data_val, labels_val, c_val)

                if n_labels<=2:
                    bce_val = f.binary_cross_entropy(recon_batch_val, data_val, reduction='sum') / batch_size
                else:
                    if model.binary_input:                        
                        bce_val = f.cross_entropy(recon_batch_val, torch.max(data_val,1)[1].type(torch.int64), reduction='sum') / batch_size
                    else:
                        bce_val = f.cross_entropy(recon_batch_val, data_val[:,0].type(torch.int64), reduction='sum') / batch_size
                kl_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp()) / batch_size
                classification_val = f.cross_entropy(logits_val, torch.argmax(labels_val, dim=1), reduction='sum') / batch_size
                accuracy_val = float(torch.sum(torch.max(logits_val[:batch_size], 1)[1].type(torch.cuda.FloatTensor) ==
                                     torch.max(labels_val, 1)[1].type(torch.cuda.FloatTensor))) / batch_size
                kl_accum_val.append(kl_val.item())
                recons_accum_val.append(bce_val.item())
                classification_accum_val.append(classification_val.item())
                accuracy_accum_val.append(accuracy_val)
            model.train()

            ## Log validation stuff
            recons_val_mean = np.mean(recons_accum_val)
            kl_val_mean = np.mean(kl_accum_val)
            class_val_mean = np.mean(classification_accum_val)
            accuracy_val_mean = np.mean(accuracy_accum_val)

            print("Validation:  rec {:.4g}  KL {:.4g}  clf {:.4g}  acc {:.4g}".format(
                  recons_val_mean, kl_val_mean, class_val_mean, accuracy_val_mean))

            writer.add_scalar('val losses/recons', recons_val_mean, model.global_step)
            writer.add_scalar('val losses/KL', kl_val_mean, model.global_step)
            writer.add_scalar('val losses/class', class_val_mean, model.global_step)
            writer.add_scalar('val losses/accuracy', accuracy_val_mean, model.global_step)


        if model.global_step % 500 == 0:

            ## Classifier output on unlabeled
            softmax_image = vutils.make_grid(softmax_u.permute(1, 0).detach())
            writer.add_image('classifier output', softmax_image, model.global_step)
            ## Save imgs
            imgs = []
            targ_size = 94
            recon_batch_u = torch.argmax(recon_batch_u,dim=1)
            if model.binary_input:
                batch_u_ = torch.argmax(batch_u,dim=1).type(torch.int64).unsqueeze(1)
            else:
                batch_u_ = batch_u.type(torch.int64)
            recon_batch_u = recon_batch_u.type(torch.int64).unsqueeze(1)
            index = np.random.randint(25, batch_u_.shape[2] - 25)
            imgs.append(pad_img(batch_u_[0:1, :, index, :, :], targ_size))
            imgs.append(pad_img(recon_batch_u[0:1, :, index, :, :], targ_size))
            index = np.random.randint(25, batch_u_.shape[3] - 25)
            imgs.append(pad_img(batch_u_[0:1, :, :, index, :], targ_size))
            imgs.append(pad_img(recon_batch_u[0:1, :, :, index, :], targ_size))
            index = np.random.randint(25, batch_u_.shape[4] - 25)
            imgs.append(pad_img(batch_u_[0:1, :, :, :, index], targ_size))
            imgs.append(pad_img(recon_batch_u[0:1, :, :, :, index], targ_size))
            #  - Concatenate and make into grid so they are displayed next to each other
            imgs = torch.cat(imgs, dim=0).detach()
            imgs = vutils.make_grid(imgs, nrow=2)
            if n_labels>2:
                imgs = to_rgb(imgs[0])
            #  - Save
            writer.add_image('images/input and recons', imgs, model.global_step)

            ## Generate samples
            #  - Sample
            with torch.no_grad():
                z = model.sample_prior(n_samples=model.y_dim)
                y = torch.arange(0, torch.tensor(model.y_dim))
                y_out = torch.zeros(model.y_dim, model.y_dim)
                y_out[torch.arange(y_out.shape[0]), y] = 1
                y = y_out
                if c_l.shape[0]!=0:
                    sample_reconstruction = model.decoder(z, y, batch_c_l[0:model.y_dim])
                else:
                    sample_reconstruction = model.decoder(z, y, None)

            #  - One slice per dimension, for all samples
            imgs = []
            sample_reconstruction = torch.argmax(sample_reconstruction,dim=1).unsqueeze(1)
            index = np.random.randint(25, batch_u_.shape[2] - 25)
            imgs.append(pad_img(sample_reconstruction[:, :, index, :, :], targ_size))
            index = np.random.randint(25, batch_u_.shape[3] - 25)
            imgs.append(pad_img(sample_reconstruction[:, :, :, index, :], targ_size))
            index = np.random.randint(25, batch_u_.shape[4] - 25)
            imgs.append(pad_img(sample_reconstruction[:, :, :, :, index], targ_size))
            #  - Concatenate and make into grid so they are displayed next to each other
            imgs = torch.cat(imgs, dim=0).detach()
            imgs = vutils.make_grid(imgs, nrow=3)
            if n_labels>2:
                imgs = to_rgb(imgs[0])
            #  - Save
            writer.add_image('generated', imgs, model.global_step)
            samples = sample_reconstruction.cpu().data.numpy()
            for class_label in range(model.y_dim):
                img = nib.Nifti1Image(samples[class_label, 0].astype(np.int8), np.eye(4))
                nib.save(img, join(logdir, "generated_class_%d_step_%d.nii.gz" % (class_label, model.global_step)))
                

    ## Save losses, avg over epoch
    writer.add_scalar('epoch losses/loss', np.mean(loss_accum), model.global_step)
    writer.add_scalar('epoch losses/recons', np.mean(recons_accum), model.global_step)
    writer.add_scalar('epoch losses/KL', np.mean(kl_accum), model.global_step)
    writer.add_scalar('epoch losses/classification', np.mean(classification_accum), model.global_step)
    writer.add_scalar('epoch losses/accuracy', np.mean(accuracy_accum), model.global_step)
    writer.add_scalar('epoch losses/categ_entropy', np.mean(h_accum), model.global_step)

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

def beta_schedule(step):
    return min(step / 6, 5000.0)   # reach 5000 at iteration 30k


def tau_schedule(step):
    decay = (np.log(tau_start) - np.log(tau_end)) / tau_steps
    tau = tau_start * np.exp(- decay * step)
    return max(tau, tau_end)

if __name__ == '__main__':
    main()
