"""
Run this script with either of the commands:
- python O2VAE_encoding.py -c <path_to_json_encoding_config>
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from O2VAE.PowderShapeAutoEncoder import full_dataloader, ShapeO2VAE



def autoencoding(config = 'O2VAE_encoding_config.json',
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Encodes input data using the encoder part of a ShapeO2VAE model.

    Args:
        config (str): Path to the JSON configuration file containing model and data loader parameters.
        device (torch.device): Device on which to run the model ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor containing the stacked latent encodings for the entire dataset.

    Notes:
        - The function assumes the existence of a ShapeO2VAE model class and a full_dataloader function.
        - The model is loaded in evaluation mode and gradients are not computed.
        - The returned tensor has shape (N, latent_dim), where N is the number of samples in the dataset.
    """
    np.float = float
    config = json.load(open(config, 'r'))
    model = ShapeO2VAE(latent_dim = config['latent_dim'],
                        enc_channels = config['enc_channels'],
                        enc_last_pooling = config['enc_last_pooling'],
                        do_sigmoid= True).to(device)
    model.load_state_dict(torch.load(config['dict_path'], weights_only= False,
                                     map_location=device))
    model.to(device)
    model.eval()

    loader = full_dataloader(config)

    encodings = []
    with torch.no_grad():
        for X in tqdm(loader):
            encodings.append(model.encoder(X.to(device))[0])
    encodings = torch.vstack(encodings)
    
    return encodings.to('cpu').numpy()

if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type = str, default = 'O2VAE_encoding_config.json',
                        help = 'Path to the config file for encoding')
    args = parser.parse_args()
    
    encodings = autoencoding(config= args.config)
    np.save('O2VAE_embeddings.npy', encodings.cpu().numpy())