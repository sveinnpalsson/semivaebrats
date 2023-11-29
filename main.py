import argparse
from datetime import datetime as dt
import os
from os.path import join
import pickle
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from utils import pad_img, print_num_params, from_np, get_data, one_hot, check_args, to_rgb, rotate_3d_batch
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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use:", device)

    logdir_suffix = create_logdir_suffix(args)
    args.models_dir = os.path.join(args.models_dir, logdir_suffix) if args.load == "" else join(args.models_dir, args.load)
    args.log_dir = os.path.join(args.log_dir, logdir_suffix)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.log_dir, 'train'))

    # Load data, determine number of labels, and clinical data dimensions
    data = get_data(args, orig_data_shape)

    # Set up the SemiVAE model
    model = setup_model(data['n_labels'], args.zdim, data['y_dim'], data['c_dim'], args.binary_input, device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Load checkpoint if a load path is provided
    if args.load != "":
        model, optimizer, start_epoch = load_model_checkpoint(args, model, optimizer)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    else:
        start_epoch = 0

    # Run training loop
    train(args, data, model, optimizer, scheduler, writer, start_epoch, device)

def train(args, data, model, optimizer, scheduler, writer, start_epoch, device):
    
    total_steps = (len(data['x_u']) // args.batch_size) * (int(args.epochs) - int(start_epoch))
    pbar = tqdm(total=total_steps, desc="Training Progress")

    for epoch in range(int(start_epoch), int(args.epochs)):
        model.train()
        metrics_epoch = []
        
        for step in range(0, len(data['x_u']), args.batch_size):
            # Prepare batch data
            batch_data = prepare_batch_data(data, args.batch_size, device, aug_flip=args.aug_flip, aug_rotate=args.aug_rotate)

            # Perform a single training step
            loss, metrics, batch_data_for_logging = train_step(model, batch_data, optimizer, args.batch_size, data['n_labels'])
            metrics_epoch.append(metrics)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix_str(f"Epoch {epoch + 1}/{args.epochs}, Step {step // args.batch_size + 1}")

            # Log training metrics at specified intervals
            if model.global_step % args.log_interval == 0:
                log_training_step(writer, metrics, model.global_step)

            if model.global_step % args.log_image_interval == 0:
                log_images(writer, model, batch_data_for_logging, model.global_step, args.log_dir, device, data['n_labels'])

            # Validation at specified intervals
            if model.global_step % args.validate_interval == 0:
                validate_model(model, data, writer, device, args.batch_size, data['n_labels'])

            # Save model checkpoint at specified intervals
            if model.global_step % args.save_interval == 0:
                save_model_checkpoint(model, optimizer, args.models_dir, epoch)

            model.global_step += 1
            scheduler.step()
            
        # Log epoch summary
        log_epoch_summary(writer, metrics_epoch, model.global_step)

    pbar.close()


def prepare_batch_data(data, batch_size, device, aug_rotate=False, aug_flip=False):
    x_u, x_l, y_l, c_l, c_u = data['x_u'], data['x_l'], data['y_l'], data['c_l'], data['c_u']

    # Randomly sample indices for labeled and unlabeled data
    batch_idx_l = np.random.choice(len(x_l), batch_size, replace=False)
    batch_idx_u = np.random.choice(len(x_u), batch_size, replace=False)

    # Extract batches for labeled and unlabeled data
    batch_x_l = torch.tensor(x_l[batch_idx_l], dtype=torch.float32, device=device)
    batch_x_u = torch.tensor(x_u[batch_idx_u], dtype=torch.float32, device=device)
    batch_y_l = torch.tensor(y_l[batch_idx_l], dtype=torch.float32, device=device)

    # Data augmentation
    if aug_rotate:
        angle = np.random.uniform(-25, 25)
        axis = np.random.choice(['x', 'y', 'z'])
        batch_x_l = rotate_3d_batch(batch_x_l, angle, axis)
        batch_x_u = rotate_3d_batch(batch_x_u, angle, axis)

    if aug_flip:
        for axis in [2, 3, 4]:  # Axis 2, 3, 4 (spatial axes)
            if np.random.rand() < 0.5:
                batch_x_l = torch.flip(batch_x_l, dims=[axis])
                batch_x_u = torch.flip(batch_x_u, dims=[axis])

    # Extract batches for clinical data if available
    batch_c_l = torch.tensor(c_l[batch_idx_l], dtype=torch.float32, device=device) if c_l.size > 0 else None
    batch_c_u = torch.tensor(c_u[batch_idx_u], dtype=torch.float32, device=device) if c_u.size > 0 else None

    return batch_x_l, batch_x_u, batch_y_l, batch_c_l, batch_c_u

# def prepare_batch_data(data, batch_size, device):
#     x_u, x_l, y_l, c_l, c_u = data['x_u'], data['x_l'], data['y_l'], data['c_l'], data['c_u']

#     # Randomly sample indices for labeled and unlabeled data
#     batch_idx_l = np.random.choice(len(x_l), batch_size, replace=False)
#     batch_idx_u = np.random.choice(len(x_u), batch_size, replace=False)

#     # Extract batches for labeled and unlabeled data
#     batch_x_l = torch.tensor(x_l[batch_idx_l], dtype=torch.float32, device=device)
#     batch_x_u = torch.tensor(x_u[batch_idx_u], dtype=torch.float32, device=device)
#     batch_y_l = torch.tensor(y_l[batch_idx_l], dtype=torch.float32, device=device)

#     # Extract batches for clinical data if available
#     batch_c_l = torch.tensor(c_l[batch_idx_l], dtype=torch.float32, device=device) if c_l.size > 0 else None
#     batch_c_u = torch.tensor(c_u[batch_idx_u], dtype=torch.float32, device=device) if c_u.size > 0 else None

#     return batch_x_l, batch_x_u, batch_y_l, batch_c_l, batch_c_u

def train_step(model, batch_data, optimizer, batch_size, n_labels):
    batch_x_l, batch_x_u, batch_y_l, batch_c_l, batch_c_u = batch_data

    # Zero the gradients before running the forward pass.
    optimizer.zero_grad()

    # Forward pass for labeled data
    recon_batch_l, mu_l, logvar_l, logits_l = model(batch_x_l, batch_y_l, batch_c_l, tau=tau_schedule(model.global_step))

    # Forward pass for unlabeled data
    recon_batch_u, mu_u, logvar_u, logits_u = model(batch_x_u, [], batch_c_u, tau=tau_schedule(model.global_step))

    # Compute losses
    loss_l, bce_l, kl_l, classification, _ = compute_loss(recon_batch_l, batch_x_l, mu_l, logvar_l, logits_l, batch_y_l, batch_size, True, n_labels, model.global_step, model.binary_input)
    loss_u, bce_u, kl_u, _, entropy = compute_loss(recon_batch_u, batch_x_u, mu_u, logvar_u, logits_u, None, batch_size, False, n_labels, model.global_step, model.binary_input)

    # Total loss
    total_loss = loss_l + loss_u

    classifier_accuracy = calculate_accuracy(logits_l, batch_y_l)

    metrics = {
        'bce_l': bce_l, 'kl_l': kl_l, 'classification': classification,
        'bce_u': bce_u, 'kl_u': kl_u, 'total_loss': total_loss, 'accuracy': classifier_accuracy,
        'entropy': entropy
    }

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    batch_data_for_logging = {}
    batch_data_for_logging['batch_u'] = batch_x_u
    batch_data_for_logging['recon_u'] = recon_batch_u
    batch_data_for_logging['logits_u'] = logits_u
    batch_data_for_logging['c_l'] = batch_c_l

    return total_loss.item(), metrics, batch_data_for_logging

def compute_loss(recon_batch, batch, mu, logvar, logits, labels, batch_size, is_labeled, n_labels, global_step, binary_input):
    # Binary cross-entropy or cross-entropy loss for reconstruction
    if n_labels <= 2:
        bce = F.binary_cross_entropy(recon_batch, batch, reduction='sum') / batch_size
    else:
        target_labels = torch.max(batch, 1)[1] if binary_input else batch[:,0]
        bce = F.cross_entropy(recon_batch, target_labels.type(torch.int64), reduction='sum') / batch_size

    # KL divergence loss
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # Classification loss and entropy calculation
    classification = 0
    entropy = 0
    if is_labeled:
        classification = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction='sum') / batch_size
    else:
        softmax_u = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(torch.mul(softmax_u, torch.log(softmax_u + 1e-12)), dim=-1).mean()

    # Combine losses
    loss = (bce + kl * beta_schedule(global_step)) / 2
    if is_labeled:
        loss += classification * alpha / 2
    else:
        loss += -entropy * gamma

    return loss, bce, kl, classification, entropy

def validate_model(model, data, writer, device, batch_size, n_labels):
    model.eval()  # Set the model to evaluation mode
    val_loss_accum = 0.0
    val_accuracy_accum = 0.0
    val_steps = 0
    x_v, y_v, c_v = data['x_v'], data['y_v'], data['c_v']

    with torch.no_grad():  # No gradients needed for validation
        for i in range(0, len(x_v), args.batch_size):
            # Prepare the validation batch
            batch_x_v = torch.tensor(x_v[i:i + args.batch_size], dtype=torch.float32, device=device)
            batch_y_v = torch.tensor(y_v[i:i + args.batch_size], dtype=torch.float32, device=device)
            batch_c_v = torch.tensor(c_v[i:i + args.batch_size], dtype=torch.float32, device=device) if c_v.size > 0 else None

            # Forward pass
            recon_batch_v, mu_v, logvar_v, logits_v = model(batch_x_v, batch_y_v, batch_c_v, tau=0.0)

            # Compute loss and accuracy
            val_loss, _, _, _, _ = compute_loss(recon_batch_v, batch_x_v, mu_v, logvar_v, logits_v, batch_y_v, batch_size, True, n_labels, model.global_step, model.binary_input)
            val_loss_accum += val_loss.item()

            val_accuracy = calculate_accuracy(logits_v, batch_y_v)
            val_accuracy_accum += val_accuracy

            val_steps += 1

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss_accum / val_steps
    avg_val_accuracy = val_accuracy_accum / val_steps

    # Log validation results
    writer.add_scalar('validation_loss', avg_val_loss, model.global_step)
    writer.add_scalar('validation_accuracy', avg_val_accuracy, model.global_step)

    model.train()  # Set the model back to training mode

def calculate_accuracy(logits, labels):
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == torch.argmax(labels, dim=1)).sum().item()
    accuracy = correct / logits.size(0)
    return accuracy

def log_training_step(writer, metrics, global_step):
    writer.add_scalar('loss/total', metrics['total_loss'], global_step)
    writer.add_scalar('loss/bce_l', metrics['bce_l'], global_step)
    writer.add_scalar('loss/kl_l', metrics['kl_l'], global_step)
    writer.add_scalar('loss/classification', metrics['classification'], global_step)
    writer.add_scalar('loss/bce_u', metrics['bce_u'], global_step)
    writer.add_scalar('loss/kl_u', metrics['kl_u'], global_step)
    writer.add_scalar('metrics/accuracy', metrics['accuracy'], global_step)
    writer.add_scalar('metrics/categorical_entropy', metrics['entropy'], global_step)

def log_epoch_summary(writer, metrics_epoch, global_step):
    # Initialize accumulators for each metric
    total_loss = bce_l = kl_l = classification = bce_u = kl_u = accuracy = h = 0

    # Aggregate metrics across all steps in the epoch
    for metrics in metrics_epoch:
        total_loss += metrics['total_loss']
        bce_l += metrics['bce_l']
        kl_l += metrics['kl_l']
        classification += metrics['classification']
        bce_u += metrics['bce_u']
        kl_u += metrics['kl_u']
        accuracy += metrics['accuracy']
        h += metrics['entropy']

    num_steps = len(metrics_epoch)

    # Calculate averages and log them
    writer.add_scalar('epoch_loss/average_total', total_loss / num_steps, global_step)
    writer.add_scalar('epoch_loss/average_bce_l', bce_l / num_steps, global_step)
    writer.add_scalar('epoch_loss/average_kl_l', kl_l / num_steps, global_step)
    writer.add_scalar('epoch_loss/average_classification', classification / num_steps, global_step)
    writer.add_scalar('epoch_loss/average_bce_u', bce_u / num_steps, global_step)
    writer.add_scalar('epoch_loss/average_kl_u', kl_u / num_steps, global_step)
    writer.add_scalar('epoch_metrics/average_accuracy', accuracy / num_steps, global_step)
    writer.add_scalar('epoch_metrics/average_categorical_entropy', h / num_steps, global_step)

def log_images(writer, model, data, global_step, logdir, device, n_labels):
    
    # Classifier output on unlabeled batch
    softmax_u = torch.softmax(data['logits_u'], dim=-1)
    softmax_image = vutils.make_grid(softmax_u.permute(1, 0).detach())
    writer.add_image('classifier_output', softmax_image, global_step)

    # Prepare images for input and reconstruction comparison
    imgs = prepare_comparison_images(data['batch_u'], data['recon_u'], model.n_labels, model.binary_input)
    imgs = vutils.make_grid(imgs, nrow=2)
    comparison_image = to_rgb(imgs[0]) if n_labels > 2 else imgs

    writer.add_image('images/input_and_recons', comparison_image, global_step)

    # Generate and log samples
    generated_imgs, samples = generate_model_samples(model, data['c_l'], device, n_labels)
    writer.add_image('generated', generated_imgs, global_step)
    save_generated_samples(samples, model.y_dim, logdir, global_step)

def get_slice_with_max_nonzero(image, axis=2):
    """
    Finds the index of the slice along the specified axis with the most nonzero content.

    Args:
    image (Tensor): The input image tensor.
    axis (int): The axis along which to find the slice. Defaults to 2 (depth).

    Returns:
    int: The index of the slice with the most nonzero content.
    """
    # Determine the dimensions to sum over
    # We sum over all dimensions except the specified axis and the batch dimension
    sum_dims = [i for i in range(len(image.shape)) if i not in [0, axis]]

    # Count the number of nonzero elements in each slice along the specified axis
    nonzero_counts = (image != 0).sum(dim=tuple(sum_dims))

    # Find the index of the slice with the maximum count
    max_index = nonzero_counts.argmax()
    return max_index.item()  # return as a Python int for indexing

def prepare_comparison_images(batch_u_, recon_batch_u, n_labels, binary_input):
    # Logic to prepare comparison images (input and reconstructed)
    targ_size = 94
    recon_batch_u = torch.argmax(recon_batch_u, dim=1)
    batch_u_ = torch.argmax(batch_u_, dim=1).unsqueeze(1) if binary_input else batch_u_.type(torch.int64)
    recon_batch_u = recon_batch_u.unsqueeze(1)

    imgs = []

    for i in range(min(len(batch_u_), 3)):  # Adjust the number of slices if needed
        index = get_slice_with_max_nonzero(batch_u_[i:i+1], axis=2)
        imgs.append(pad_img(batch_u_[i:i+1, :, index, :, :], targ_size))
        imgs.append(pad_img(recon_batch_u[i:i+1, :, index, :, :], targ_size))

    return torch.cat(imgs, dim=0).detach()

def generate_model_samples(model, c_l, device, n_labels):
    targ_size = 94
    with torch.no_grad():
        z = model.sample_prior(n_samples=model.y_dim).to(device)
        y = torch.eye(model.y_dim).to(device)
        sample_reconstruction = model.decoder(z, y, c_l[0:model.y_dim]) if (c_l is not None and c_l.shape[0] != 0) else model.decoder(z, y, None)
        sample_reconstruction = torch.argmax(sample_reconstruction, dim=1).unsqueeze(1)
        imgs = []
        index = get_slice_with_max_nonzero(sample_reconstruction[0], 1)
        imgs.append(pad_img(sample_reconstruction[0:1, :, index, :, :], targ_size))
        index = get_slice_with_max_nonzero(sample_reconstruction[0], 2)
        imgs.append(pad_img(sample_reconstruction[0:1, :, :, index, :], targ_size))
        index = get_slice_with_max_nonzero(sample_reconstruction[0], 3)
        imgs.append(pad_img(sample_reconstruction[0:1, :, :, :, index], targ_size))
        imgs = torch.cat(imgs, dim=0).detach()
        imgs = vutils.make_grid(imgs, nrow=3)
        if n_labels>2:
            imgs = to_rgb(imgs[0])

        return imgs, sample_reconstruction.cpu().data.numpy()

def save_generated_samples(samples, y_dim, logdir, global_step):
    for class_label in range(y_dim):
        img = nib.Nifti1Image(samples[class_label, 0].astype(np.int8), np.eye(4))
        nib.save(img, os.path.join(logdir, f"generated_class_{class_label}_step_{global_step}.nii.gz"))

def save_model_checkpoint(model, optimizer, models_dir, epoch):
    checkpoint_path = os.path.join(models_dir, f'model_checkpoint_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': model.global_step
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

def load_model_checkpoint(args, model, optimizer):
    print("Loading model from %s" % args.models_dir)
    nums = [int(i.split("_")[-1].replace('.pt', '')) for i in os.listdir(args.models_dir)]
    start_epoch = max(nums)
    model_path = join(args.models_dir, f"model_checkpoint_{start_epoch}.pt")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    model.global_step = checkpoint['global_step']
    print("Loaded model at epoch %d, total steps: %d" % (start_epoch, model.global_step))
    return model, optimizer, start_epoch

def setup_model(n_labels, z_dim, y_dim, c_dim, binary_input, device):
    model = SemiVAE(z_dim, y_dim, c_dim, n_labels=n_labels, binary_input=binary_input).to(device)
    print_num_params(model)
    return model

def create_logdir_suffix(args):
    logdir_suffix = '-{}-zdim={}-beta=5000-alpha={:.5f}-lr={:.5f}-gamma={}-batch={}'.format(
        args.data_dir_train.replace("Train", "").replace(".", "").replace("/", ""),
        args.zdim, alpha, args.learning_rate, gamma, args.batch_size
    )
    if args.use_age:
        logdir_suffix += "-age"
    if args.use_rs:
        logdir_suffix += "-rs"
    if args.use_mgmt:
        logdir_suffix += "-mgmt"
    if args.binary_input:
        logdir_suffix += "-binary_input"

    return str(dt.now())[:-7].replace(":", "-").replace(" ", "-") + logdir_suffix


def beta_schedule(step):
    return min(step / 6, 5000.0)   # reach 5000 at iteration 30k


def tau_schedule(step):
    decay = (np.log(tau_start) - np.log(tau_end)) / tau_steps
    tau = tau_start * np.exp(- decay * step)
    return max(tau, tau_end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/brats_20_semivae_dataset', metavar='DATADIR', help="data directory, created with the create_dataset script")    
    parser.add_argument('--log_dir', type=str, default='logs/', metavar='LOGS', help="logs directory")
    parser.add_argument('--models_dir', type=str, default='models/', metavar='MODELS',  help="models directory")
    parser.add_argument('--batch_size', type=int, default=16, metavar='BATCH', help="batch size")
    parser.add_argument('--learning_rate', type=float, default=2.0e-5, metavar='LR', help="learning rate")
    parser.add_argument('--epochs', type=int, default=20000, metavar='EPOCHS', help="number of epochs")
    parser.add_argument('--zdim', type=int, default=16, metavar='ZDIM', help="Number of dimensions in latent space")
    parser.add_argument('--log_interval', type=int, default=10, metavar='LOGINT', help="Training steps between logging")
    parser.add_argument('--log_image_interval', type=int, default=500, metavar='LOGIMAGEINT', help="Training steps between logging images")
    parser.add_argument('--validate_interval', type=int, default=20, metavar='VALINT', help="Training steps between model validation")
    parser.add_argument('--save_interval', type=int, default=2000, metavar='SAVEINT', help="Training steps between model validation")
    parser.add_argument('--load', type=str, default='', metavar='LOADDIR', help="time string of previous run to load from")
    parser.add_argument('--binary_input', action='store_true', help="Set this flag to use one input channel for each tumor structure")
    parser.add_argument('--use_age', action='store_true', help="Set this flag to use age in prediction")
    parser.add_argument('--use_rs', action='store_true', help="Set this flag to use resection status in prediction")
    parser.add_argument('--use_mgmt', action='store_true', help="Set this flag to use MGMT value in prediction")
    parser.add_argument('--aug_rotate', action='store_true', help="Set this flag to use rotation augmentation")
    parser.add_argument('--aug_flip', action='store_true', help="Set this flag to use flip augmentation")

    args = parser.parse_args()
    args.data_dir_train = join(args.data_dir, 'Train')
    args.data_dir_val = join(args.data_dir, 'Validation')
    args.data_info_path = join(args.data_dir, 'info_table.csv')
    main(args)