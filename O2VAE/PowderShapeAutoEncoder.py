import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as v2

from .e2scnn import E2SFCNN as o2_encoder
from .model_utils import sample_gaussian, O2Loss

### Custom Powder Dataset
def pad_and_resize(image, target_size):
        """
        Pads an image with zeros to make it square and resizes it to the target size.
        
        Args:
            image: Input image.
            target_size (int): Desired output size (height and width will be equal).

        Returns:
            Resized square image.
        """
        # Get original dimensions
        _, height, width = image.shape
        
        # Determine padding
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        
        # Define padding transform
        extra = max(width, height) // 4
        pad_transform = v2.Pad((pad_left + extra, pad_top + extra, pad_right + extra, pad_bottom + extra), fill=0, padding_mode='constant')
        
        # Pad the image
        padded_image = pad_transform(image)
        
        # Resize to target size
        resize_transform = v2.Resize((target_size, target_size))
        resized_image = resize_transform(padded_image)

        return resized_image

class PowderDataset(Dataset):

    def __init__(self, datapath, transform = pad_and_resize, img_size = 128, ext = '.bmp'):
        self.datapath = datapath
        self.img_size = img_size
        self.image_paths = [path for path in os.listdir(self.datapath) if path.endswith(ext)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #read image
        imgpath = os.path.join(self.datapath, self.image_paths[idx])
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = torch.from_numpy(cv2.bitwise_not(img))[None, :, :].float()
        
        if self.transform:
             img = self.transform(img, self.img_size)

        img = torch.where(img > 0, torch.tensor(255, dtype=img.dtype, device=img.device), img)
        img /= 255.0

        return img

### Dataloaders for training, validation and testing
def make_dataloaders(config):
    '''
    batch_size: int = batch size
    ntimesteps: int or list = number/list of time steps in data
    nsample: int = number of samples to use
    split: list = list of train set to data ration and train+valid set to data ratio
    '''
    #initialize dataset object
    dataset = PowderDataset(datapath = config['datapath'],
                            ext = config['ext'],
                            img_size = config['img_size'])

    # Creating Training, Validation, and Test dataloaders
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_split = int(np.floor(config['split'][0] * dataset_size))
    val_split = int(np.floor(config['split'][1] * dataset_size))
    shuffle_dataset = True
    random_seed = 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[ : train_split]
    val_indices = indices[train_split : train_split + val_split]
    test_indices = indices[train_split + val_split : ]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=train_sampler)#, num_workers = os.cpu_count()//2 - 2)
    validation_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=valid_sampler)#, num_workers = os.cpu_count()//2 - 2)
    test_loader = DataLoader(dataset, batch_size = config['batch_size'], sampler=test_sampler)#, num_workers = os.cpu_count()//2 - 2)

    return train_loader, validation_loader, test_loader

## Dataloader for loading entired dataset
def full_dataloader(config):
    '''
    batch_size: int = batch size
    ntimesteps: int or list = number/list of time steps in data
    nsample: int = number of samples to use
    split: list = list of train set to data ration and train+valid set to data ratio
    '''
    #initialize dataset object
    dataset = PowderDataset(datapath = config['datapath'],
                            ext = config['ext'],
                            img_size = config['img_size'])

    return DataLoader(dataset, batch_size = config['batch_size'], shuffle = False, num_workers= 0)


class ShapeO2VAE(torch.nn.Module):
    def __init__(self, latent_dim = 20,
                 in_channels = 1,
                 enc_channels = [6, 9, 12, 12, 19, 25],
                 enc_last_pooling = "pointwise_adaptive_max_pool",
                 do_sigmoid = False) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.do_sigmoid = do_sigmoid
        self.enc_channels = enc_channels
        self.in_channels = in_channels
        self.enc_last_pooling = enc_last_pooling

        self.decoder = torch.nn.Sequential(
                                torch.nn.ELU(),
                                torch.nn.Linear(self.latent_dim, 128),
                                torch.nn.ELU(),
                                torch.nn.Linear(128, 4096),
                                torch.nn.ELU(),
                                torch.nn.Unflatten(1, (64, 8, 8)),

                                torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                torch.nn.BatchNorm2d(32),
                                torch.nn.ELU(),

                                torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                                torch.nn.BatchNorm2d(16),
                                torch.nn.ELU(),

                                torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                                torch.nn.BatchNorm2d(8),
                                torch.nn.ELU(),

                                torch.nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
                                torch.nn.BatchNorm2d(1),
                                torch.nn.ELU(),

                                torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
                                # torch.nn.Sigmoid()  # Uncomment if using BCE loss
                            )

        self.enc_model = o2_encoder(n_channels= self.in_channels,
                       n_classes = self.latent_dim * 2,
                       cnn_dims= self.enc_channels,
                       last_pooling= self.enc_last_pooling)
        
    def encoder(self, x):
        x = self.enc_model(x)
        mu, logvar = torch.split(x, self.latent_dim, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return sample_gaussian(mu, logvar)

    def loss(self, out, x, mu, logvar, loss_config):
        recon_loss = O2Loss(image_shape= (loss_config['img_size'], loss_config['img_size']),
                                loss_fun= loss_config['recon_loss_fun'])(out, x)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        vae_loss = recon_loss + loss_config['beta'] * kld_loss

        return vae_loss, recon_loss, kld_loss

    def forward(self, x):
        mu, logvar = self.encoder(x) #latent representation vector of length latent_dim
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)

        if self.do_sigmoid:
            out = torch.sigmoid(out)
        return out, mu, logvar