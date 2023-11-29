import torch
import torch.nn.functional as F
from torch import nn

from utils import Crop3d, Interpolate, Reshape, print_shape

use_selu = True

def get_nonlin(channels, use_selu=True):
    if use_selu:
        return nn.SELU(),
    return nn.BatchNorm3d(channels), nn.ReLU()

def create_block(in_channels, out_channels, kernel_size, stride, padding, dropout_rate, use_groups=False):
    groups = out_channels if use_groups else 1
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups),
        *get_nonlin(out_channels),
        nn.Dropout3d(dropout_rate)
    )

def create_skip_block(in_channels, out_channels, kernel_size, stride, padding, pool_kernel, pool_stride, pool_padding):
    return nn.Sequential(
        nn.AvgPool3d(pool_kernel, stride=pool_stride, padding=pool_padding),
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        Crop3d(0, 1, 0)
    )

class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim, embed_y, y_embed_size, c_dim, embed_c, c_embed_size, input_channels):
        super(Encoder, self).__init__()
        self.first_pass = True
        self.input_channels = input_channels

        self.embed_y = embed_y
        self.embed_c = embed_c
        self.c_embed_size = c_embed_size
        self.y_embed_size = y_embed_size

        self.layer1 = nn.Sequential(
            nn.Conv3d(self.input_channels, 16, 7, stride=4, padding=4),
            *get_nonlin(16),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, 5, stride=2, padding=2, groups=2),
            *get_nonlin(32),
            nn.Dropout3d(0.2),
            nn.Conv3d(32, 64, 5, padding=2, groups=4),
            nn.Dropout3d(0.2),
        )
        self.block2_skip = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=1),
            nn.Conv3d(16, 64, 3, padding=1),
            Crop3d(0, 1, 0),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 32, 5, stride=2, padding=2, groups=4),
            *get_nonlin(32),
            nn.Conv3d(32, 128, 5, stride=2, padding=2, groups=8),
        )
        self.block3_skip = nn.Sequential(
            nn.AvgPool3d(4, stride=4),
            nn.Conv3d(64, 128, 1, padding=0),
            nn.ConstantPad3d((1, 0, 0, 0, 1, 0), 0.0),
        )

        self.last_conv = nn.Sequential(
            nn.Conv3d(128 + y_embed_size + c_embed_size, z_dim * 2, 1),
            nn.AvgPool3d(3)
            )

        # CLASSIFIER (copy architecture of encoder, different parameters)
        self.classifier_block2 = nn.Sequential(
            nn.Conv3d(16, 32, 5, stride=2, padding=2, groups=4),
            *get_nonlin(32),
            nn.Dropout3d(0.5),
            nn.Conv3d(32, 64, 5, padding=2, groups=8),
            nn.Dropout3d(0.5),
        )
        self.classifier_block2_skip = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=1),
            nn.Conv3d(16, 64, 3, padding=1),
            nn.Dropout3d(0.5),
            Crop3d(0, 1, 0),
        )

        self.classifier_block3 = nn.Sequential(
            nn.Conv3d(64, 32, 5, stride=2, padding=2, groups=8),
            *get_nonlin(32),
            nn.Dropout3d(0.5),
            nn.Conv3d(32, 128, 5, stride=2, padding=2, groups=16),
        )
        self.classifier_block3_skip = nn.Sequential(
            nn.AvgPool3d(4, stride=4),
            nn.Conv3d(64, 128, 1, padding=0),
            nn.Dropout3d(0.5),
            nn.ConstantPad3d((1, 0, 0, 0, 1, 0), 0.0),
        )
        self.classifier_combine_paths = nn.Conv3d(256+c_embed_size, 128, 1)
        self.classifier_logits = nn.Sequential(  # in: (128, 3, 3, 3)
            nn.Dropout3d(0.5),
            nn.Conv3d(128, 64, 3, groups=4),   # (64, 1, 1, 1)
            *get_nonlin(64),
            Reshape(64),
            nn.Dropout(0.5),
            nn.Linear(64, y_dim)
        )

    def forward(self, x, y=None, c=None, tau=0.0, hard=False, classify=False):
        # Debug print for the first pass
        if self.first_pass:
            print_shape(x)  # Initial shape of x

        # First layer
        x = self.layer1(x)
        x_clf = x

        # Encoder Path
        x = self._apply_encoder_path(x)

        # Classifier Path
        logits, x_clf = self._apply_classifier_path(x, x_clf, c)

        # If classify mode is enabled, return early with logits
        if classify:
            return logits

        # Handle labels
        y_s = self._handle_labels(y, logits, tau, hard)

        # Merge paths and apply the last convolution
        x = self._merge_paths(x, y_s, c)

        # Update first pass flag
        self.first_pass = False

        return x, logits, y_s

    def _apply_encoder_path(self, x):
        x = F.leaky_relu(self.block2(x) + self.block2_skip(x))
        x = F.leaky_relu(self.block3(x) + self.block3_skip(x))
        return x

    def _apply_classifier_path(self, x, x_clf, c):
        x_clf = F.leaky_relu(self.classifier_block2(x_clf) + self.classifier_block2_skip(x_clf))
        x_clf = F.leaky_relu(self.classifier_block3(x_clf) + self.classifier_block3_skip(x_clf))

        if self.c_embed_size != 0 and c is not None:
            c_embedded = self._embed_and_expand(c, x.shape[2:], self.embed_c)
            x_clf = self.classifier_combine_paths(torch.cat([x, x_clf, c_embedded], dim=1))
        else:
            x_clf = self.classifier_combine_paths(torch.cat([x, x_clf], dim=1))

        logits = self.classifier_logits(x_clf)
        return logits, x_clf

    def _handle_labels(self, y, logits, tau, hard):
        if y is None or len(y) == 0:  # Unlabeled batch
            return F.gumbel_softmax(logits, tau, hard=hard)
        else:
            return y  # Use true labels

    def _embed_and_expand(self, tensor, shape, embed):
        embedded = embed(tensor)
        embedded = embedded.view(embedded.shape[0], embedded.shape[1], 1, 1, 1)
        return embedded.expand(-1, -1, *shape)

    def _merge_paths(self, x, y_s, c):
        y_embedded = self._embed_and_expand(y_s, x.shape[2:], self.embed_y)

        if self.c_embed_size != 0:
            c_embedded = self._embed_and_expand(c, x.shape[2:], self.embed_c)
            return self.last_conv(torch.cat([x, y_embedded, c_embedded], dim=1))
        else:
            return self.last_conv(torch.cat([x, y_embedded], dim=1))

    
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim, embed_y, y_embed_size, embed_c, c_embed_size, n_labels=2):
        super().__init__()
        self.first_pass = True
        self.n_labels = n_labels
        self.embed_y = embed_y
        self.embed_c = embed_c

        self.layer_1 = nn.Sequential(
            Interpolate(scale=3),
            nn.Conv3d(z_dim, z_dim, 3, padding=1, groups=z_dim // 8),
            nn.LeakyReLU(negative_slope=0.05),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv3d(z_dim + y_embed_size + c_embed_size, 128 - y_embed_size - c_embed_size, 1),
            *get_nonlin(128)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, groups=2),
            *get_nonlin(64),
            nn.ConvTranspose3d(64, 32, 5, stride=2, padding=1, groups=4),
        )
        self.block4_skip = nn.Sequential(
            nn.Conv3d(128, 32, 1, padding=0),
            Interpolate(scale=4),
            Crop3d(1, 1, 1),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose3d(32 + y_embed_size + c_embed_size, 16, 3, stride=2, padding=1),
            *get_nonlin(64),
            nn.ConvTranspose3d(16, 16, 5, stride=2, padding=2, groups=2, output_padding=(0, 1, 1)),
            *get_nonlin(64),
            nn.Conv3d(16, 16, 5, padding=2),
        )
        self.block3_skip = nn.Sequential(
            Interpolate(scale=2),
            nn.Conv3d(32 + y_embed_size + c_embed_size, 16, 1, padding=0),
            Interpolate(scale=2),
            Crop3d(3, 2, 2),
            nn.Conv3d(16, 16, 1, padding=0),
        )
        if self.n_labels <= 2:
            self.final_block = nn.Sequential(
                nn.Conv3d(16, 1, 5, padding=2),
                Interpolate(scale=2, mode='trilinear'),
            )
        else:
            self.final_block = nn.Sequential(
                nn.Conv3d(16,self.n_labels, 5, padding=2),
                Interpolate(scale=2, mode='trilinear'),
            )


    def forward(self, x, y, c):
        # Embed label and clinical feature tensors
        y_embedded = self.embed_y(y)
        c_embedded = self.embed_c(c) if c is not None else None

        # Process through layer_1
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        x = self.layer_1(x)

        # Expand and concatenate for layer_2
        x = self._concat_expand(x, y_embedded, c_embedded)
        x = self.layer_2(x)

        # Expand and concatenate for block4
        x = self._concat_expand(x, y_embedded, c_embedded)
        x = self.block4(x) + self.block4_skip(x)
        x = F.leaky_relu(x)
        x = F.pad(x, [0, 0, 1, 1, 0, 0])
        x = x[:, :, 1:, :, 1:-1]

        # Expand and concatenate for block3
        x = self._concat_expand(x, y_embedded, c_embedded)
        x = self.block3(x) + self.block3_skip(x)
        x = x[:, :, :, 1:-1, 1:-1]
        x = F.leaky_relu(x)

        # Apply final block
        x = self.final_block(x)
        x = x[:, :, 1:, 1:-1, :]
        x = torch.sigmoid(x) if self.n_labels <= 2 else x

        self.first_pass = False
        return x

    def _concat_expand(self, x, y_embedded, c_embedded):
        # Expand y and c to match the dimensions of x
        y_expanded = y_embedded.view(y_embedded.shape[0], -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        if c_embedded is not None:
            c_expanded = c_embedded.view(c_embedded.shape[0], -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
            return torch.cat([x, y_expanded, c_expanded], dim=1)
        return torch.cat([x, y_expanded], dim=1)


class SemiVAE(nn.Module):
    def __init__(self, z_dim, y_dim, c_dim, n_labels=2, binary_input=False):
        super(SemiVAE, self).__init__()
        self.global_step = 0
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.n_labels = n_labels
        self.binary_input = binary_input

        enc_input_channels = n_labels if binary_input else 1

        self.y_embed_size = 8
        self.embed_y = nn.Linear(y_dim, self.y_embed_size)

        self.c_embed_size = 16 if c_dim != 0 else 0
        self.embed_c = nn.Linear(c_dim, self.c_embed_size) if c_dim != 0 else None

        self.encoder = Encoder(z_dim, y_dim, self.embed_y, self.y_embed_size,
                               c_dim, self.embed_c, self.c_embed_size, enc_input_channels)
        self.decoder = Decoder(z_dim, self.embed_y, self.y_embed_size, self.embed_c, self.c_embed_size, n_labels=n_labels)

    def classify(self, x, c):
        return self.encoder(x, [], c, classify=True)

    def sample(self, mu, logvar):
        assert mu.size(1) == logvar.size(1) == self.z_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_prior(self, n_samples=1):
        return self.sample(torch.zeros(n_samples, self.z_dim), torch.zeros(n_samples, self.z_dim))

    def forward(self, x, y, c, tau=None, sample=True):
        mulogvar, logits, y_s = self.encoder(x, y, c, tau=tau)
        mu, logvar = mulogvar.chunk(2, 1)
        z_s = self.sample(mu, logvar) if sample else mu
        recons = self.decoder(z_s, y_s, c)
        return recons, mu, logvar, logits

def get_nonlin(channels):
    # Return tuple to be unpacked
    if use_selu:
        return nn.SELU(),
    return nn.BatchNorm3d(channels), nn.ReLU()
