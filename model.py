import torch
import torch.nn.functional as f
from torch import nn

from utils import Crop3d, Interpolate, Reshape, print_shape

use_selu = True

class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim, embed_y, y_embed_size, c_dim, embed_c, c_embed_size, input_channels):
        super().__init__()
        self.first_pass = True
        self.input_channels = input_channels

        self.embed_y = embed_y
        self.embed_c = embed_c
        self.c_embed_size = c_embed_size
        self.y_embed_size = y_embed_size

        self.layer1 = nn.Sequential(
            nn.Conv3d(self.input_channels, 16, 7, stride=4, padding=4),
            # nn.Conv3d(1, 32, 8, stride=4, padding=4),
            *get_nonlin(16),
        )


        self.block2 = nn.Sequential(
            nn.Conv3d(16, 64, 5, stride=2, padding=2, groups=1),
            *get_nonlin(64),
            nn.Dropout3d(0.2),
            nn.Conv3d(64, 64, 5, padding=2, groups=2),
            nn.Dropout3d(0.2),
        )
        self.block2_skip = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=1),
            nn.Conv3d(16, 64, 3, padding=1),
            Crop3d(0, 1, 0),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 32, 5, stride=2, padding=2, groups=2),
            *get_nonlin(32),
            nn.Conv3d(32, 128, 5, stride=2, padding=2, groups=4),
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

        ##### CLASSIFIER (copy architecture of encoder, different parameters)
        self.classifier_block2 = nn.Sequential(
            nn.Conv3d(16, 64, 5, stride=2, padding=2, groups=1),
            *get_nonlin(64),
            nn.Dropout3d(0.5),
            nn.Conv3d(64, 64, 5, padding=2, groups=2),
            nn.Dropout3d(0.5),
        )
        self.classifier_block2_skip = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=1),
            nn.Conv3d(16, 64, 3, padding=1),
            nn.Dropout3d(0.5),
            Crop3d(0, 1, 0),
        )

        self.classifier_block3 = nn.Sequential(
            nn.Conv3d(64, 32, 5, stride=2, padding=2, groups=2),
            *get_nonlin(32),
            nn.Dropout3d(0.5),
            nn.Conv3d(32, 128, 5, stride=2, padding=2, groups=4),
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
        #########


    def forward(self, x, y, c, tau=None, hard=False, classify=False):
        print_shape(x, self.first_pass)

        x = self.layer1(x)
        x_c = x
        print_shape(x, self.first_pass)   # (19, 24, 17)

        # print(self.block2(x).shape, self.block2_skip(x).shape)
        x = self.block2(x) + self.block2_skip(x)
        x = f.leaky_relu(x)
        print_shape(x, self.first_pass)   # (10, 12, 9)

        # print(self.block3(x).shape, self.block3_skip(x).shape)
        x = self.block3(x) + self.block3_skip(x)
        x = f.leaky_relu(x)
        print_shape(x, self.first_pass)   # (3, 3, 3)

        #### CLASSIFIER
        x_c = self.classifier_block2(x_c) + self.classifier_block2_skip(x_c)
        x_c = f.leaky_relu(x_c)

        x_c = self.classifier_block3(x_c) + self.classifier_block3_skip(x_c)
        x_c = f.leaky_relu(x_c)
        if self.c_embed_size!=0:
            c_embedded = self.embed_c(c)
            c_embedded = c_embedded.view(c_embedded.shape[0], c_embedded.shape[1],1,1,1)
            c_embedded = c_embedded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])

            ## As input to classifier, use output of specialized path as well as shared path with encoder
            x_c = self.classifier_combine_paths(torch.cat([x, x_c, c_embedded], dim=1))
        else:
            x_c = self.classifier_combine_paths(torch.cat([x, x_c], dim=1))

        logits = self.classifier_logits(x_c)
        if classify:
            return logits
        if len(y) == 0:  # unlabeled batch
            y_s = f.gumbel_softmax(logits, tau, hard=hard)  # gumbel-softmax samples (same size as logits)
        else:
            y_s = y   # use true labels
        ############

        # Make ydim one-hot channels (or embeddings) with same volume dims as x
        y_embedded = self.embed_y(y_s)
        y_embedded = y_embedded.view(y_embedded.shape[0], y_embedded.shape[1], 1, 1, 1)
        y_embedded = y_embedded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        
        if self.c_embed_size !=0:
            x = torch.cat([x, y_embedded, c_embedded], dim=1)  # concat along channel dimension
        else:
            x = torch.cat([x, y_embedded], dim=1)              
        x = self.last_conv(x)

        self.first_pass = False

        return x, logits, y_s    
    
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim, embed_y, y_embed_size, embed_c, c_embed_size, n_labels=2):
        super().__init__()
        self.first_pass = True
        self.n_labels = n_labels
        self.embed_y = embed_y
        self.embed_c = embed_c

        ## no-FC version: from zdim x 1 x 1 x 1 to 128 x 3 x 3 x 3
        self.layer_1 = nn.Sequential(
            Interpolate(scale=3),
            nn.Conv3d(z_dim, z_dim, 3, padding=1, groups=z_dim // 8),
            nn.LeakyReLU(negative_slope=0.05),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv3d(z_dim + y_embed_size + c_embed_size, 128, 1),
            *get_nonlin(128)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose3d(128 + y_embed_size + c_embed_size, 64, 3, stride=2, padding=1),
            *get_nonlin(64),
            nn.ConvTranspose3d(64, 128, 5, stride=2, padding=1, groups=4),
        )
        self.block4_skip = nn.Sequential(
            nn.Conv3d(128 + y_embed_size + c_embed_size, 128, 1, padding=0),
            Interpolate(scale=4),
            Crop3d(1, 1, 1),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose3d(128 + y_embed_size + c_embed_size, 64, 3, stride=2, padding=1),
            *get_nonlin(64),
            nn.ConvTranspose3d(64, 64, 5, stride=2, padding=2, groups=2, output_padding=(0, 1, 1)),
            *get_nonlin(64),
            nn.Conv3d(64, 16, 5, padding=2),
        )
        self.block3_skip = nn.Sequential(
            Interpolate(scale=2),
            nn.Conv3d(128 + y_embed_size + c_embed_size, 32, 1, padding=0),
            Interpolate(scale=2),
            Crop3d(3, 2, 2),
            nn.Conv3d(32, 16, 1, padding=0),
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
        y_embedded = self.embed_y(y)
        if c is not None:
            c_embedded = self.embed_c(c)
        else:
            c_embedded = None

        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        print_shape(x, self.first_pass)
        x = self.layer_1(x)
        print_shape(x, self.first_pass)

        y_expanded = y_embedded.view(y_embedded.shape[0], y_embedded.shape[1], 1, 1, 1)
        y_expanded = y_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        if c_embedded is not None:
            c_expanded = c_embedded.view(c_embedded.shape[0], c_embedded.shape[1], 1, 1, 1)
            c_expanded = c_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
            x = self.layer_2(torch.cat([x, y_expanded, c_expanded], dim=1))
        else:
            x = self.layer_2(torch.cat([x, y_expanded], dim=1))            

        # print(self.block4(x).shape, self.block4_skip(x).shape)
        y_expanded = y_embedded.view(y_embedded.shape[0], y_embedded.shape[1], 1, 1, 1)
        y_expanded = y_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        if c_embedded is not None:        
            c_expanded = c_embedded.view(c_embedded.shape[0], c_embedded.shape[1], 1, 1, 1)
            c_expanded = c_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
            x_y = torch.cat([x, y_expanded, c_expanded], dim=1)
        else:
            x_y = torch.cat([x, y_expanded], dim=1)

        x = self.block4(x_y) + self.block4_skip(x_y)

        x = f.leaky_relu(x)
        x = f.pad(x, [0, 0, 1, 1, 0, 0])
        x = x[:, :, 1:, :, 1:-1]
        print_shape(x, self.first_pass)

        # print(self.block3(x).shape, self.block3_skip(x).shape)
        y_expanded = y_embedded.view(y_embedded.shape[0], y_embedded.shape[1], 1, 1, 1)
        y_expanded = y_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        if c_embedded is not None:        
            c_expanded = c_embedded.view(c_embedded.shape[0], c_embedded.shape[1], 1, 1, 1)
            c_expanded = c_expanded.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
            x_y = torch.cat([x, y_expanded, c_expanded], dim=1)
        else:
            x_y = torch.cat([x, y_expanded], dim=1)
            
        x = self.block3(x_y)[:, :, :, :, :] + self.block3_skip(x_y)
        x = x[:, :, :, 1:-1, 1:-1]
        x = f.leaky_relu(x)
        print_shape(x, self.first_pass)

        x = self.final_block(x)
        x = x[:, :, 1:, 1:-1, :]
        if self.n_labels<=2:
            x = torch.sigmoid(x)
        else:
            #x = torch.softmax(x,dim=1)
            x=x
        print_shape(x, self.first_pass)

        self.first_pass = False
        return x



class SemiVAE(nn.Module):

    def __init__(self, z_dim, y_dim, c_dim, n_labels=2,binary_input=False):
        super().__init__()
        self.global_step = 0
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.binary_input = binary_input
        if binary_input:
            enc_input_channels = n_labels
        else:
            enc_input_channels = 1

        self.y_embed_size = 8
        self.embed_y = nn.Sequential(
#            nn.Linear(y_dim, 32),
#            nn.LeakyReLU(),
#            nn.Linear(32, self.y_embed_size)
            nn.Linear(y_dim,self.y_embed_size)
        )
        if c_dim!=0:
            self.c_embed_size = 16
            self.embed_c = nn.Sequential(
#                nn.Linear(c_dim, 32),
#                nn.LeakyReLU(),
#                nn.Linear(32, self.c_embed_size)
                nn.Linear(c_dim,self.c_embed_size)
            )
        else:
            self.embed_c = None
            self.c_embed_size = 0

        self.encoder = Encoder(z_dim, self.y_dim, self.embed_y, self.y_embed_size,
                               self.c_dim, self.embed_c, self.c_embed_size, enc_input_channels)
        self.decoder = Decoder(z_dim, self.embed_y, self.y_embed_size, self.embed_c, self.c_embed_size, n_labels=n_labels)
        self.n_labels = n_labels

    def classify(self,x,c):
        return self.encoder(x, [], c, classify=True)

        
    def sample(self, mu, logvar):  # TODO y
        assert mu.size(1) == logvar.size(1) == self.z_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_prior(self, n_samples=1):
        return self.sample(torch.zeros(n_samples, self.z_dim), torch.zeros(n_samples, self.z_dim))

    def forward(self, x, y, c, tau=None,sample=True):   # y is either a tensor of labels or an empty list
        mulogvar, logits, y_s = self.encoder(x, y, c, tau=tau)  # returned y_s are either true labels if given, or samples from q(y | x)
        mu, logvar = mulogvar.chunk(2, 1)
        if sample:
            z_s = self.sample(mu, logvar)
        else:
            z_s = mu
        recons = self.decoder(z_s, y_s, c)
        return recons, mu, logvar, logits


def get_nonlin(channels):
    # Return tuple to be unpacked
    if use_selu:
        return nn.SELU(),
    return nn.BatchNorm3d(channels), nn.ReLU()
