import torch
import torch.nn as nn
import torch.nn.functional as F

# Base VAE class containg common methods for all other VAE classes
class BaseVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes, temp=0.67):
        super(BaseVariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.temp = temp
# function to reparametrise the latent sampling process
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def gumbel_softmax(self, logits):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
        y_sampled = F.softmax((logits + gumbel_noise) / self.temp, dim=1)
        return y_sampled

# class to define VAE for MNIST (1 channel, 28x28 images)
class MNISTVariationalAutoEncoder(BaseVariationalAutoEncoder):
    def __init__(self, latent_dim, num_classes, temp=0.67):
        super(MNISTVariationalAutoEncoder, self).__init__(latent_dim, num_classes, temp)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)
        self.fc_logits = nn.Linear(512, num_classes)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x, y_onehot=None):
        # passing input to the encoder
        e = self.encoder(x)
        mean = self.fc_mean(e)
        log_var = self.fc_log_var(e)
        logits = self.fc_logits(e)

        # Sampling from the latent variables
        z = self.reparameterize(mean, log_var)
        '''
        if y_onehot is not None:  # for labeled case
            z_combined = torch.cat([z, y_onehot], dim=1)
        else:  # for  Unlabeled case
            y_sampled = self.gumbel_softmax(logits)
            z_combined = torch.cat([z, y_sampled], dim=1)
        '''
        if y_onehot is None:  
              # for  Unlabeled case
            y_sampled = self.gumbel_softmax(logits)
            z_combined = torch.cat([z, y_sampled], dim=1)
        else: # for labeled case
            z_combined = torch.cat([z, y_onehot], dim=1)

        # Decode
        z_combined = self.decoder_input(z_combined)
        reconstructed = self.decoder(z_combined)

        return reconstructed, mean, log_var, logits

# Building VAE for CIFAR-10 (3 channels, 32x32 images)
class CIFAR10VariationalAutoEncoder(BaseVariationalAutoEncoder):
    def __init__(self, latent_dim, num_classes, temp=0.67):
        super(CIFAR10VariationalAutoEncoder, self).__init__(latent_dim, num_classes, temp)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),nn.ReLU(),
        )
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)
        self.fc_logits = nn.Linear(512, num_classes)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x, y_onehot=None):
        # pass the input to the Encoder
        e = self.encoder(x)
        mean = self.fc_mean(e)
        log_var = self.fc_log_var(e)
        logits = self.fc_logits(e)

        # Sampling latent variables
        z = self.reparameterize(mean, log_var)

        if y_onehot is None:  
              # for  Unlabeled case
            y_sampled = self.gumbel_softmax(logits)
            z_combined = torch.cat([z, y_sampled], dim=1)
        else: # for labeled case
            z_combined = torch.cat([z, y_onehot], dim=1)


        # Decode
        z_combined = self.decoder_input(z_combined)
        reconstructed = self.decoder(z_combined)

        return reconstructed, mean, log_var, logits


