import torch
import numpy as np
from O2VAE.PowderShapeAutoEncoder import full_dataloader, ShapeO2VAE
from tqdm import tqdm


def autoencoding(model, config, device, vae : bool = False):
    model.to(device)
    model.eval()
    loader = full_dataloader(config)
    encodings = []

    with torch.no_grad():
        for X in tqdm(loader):
            if not vae:
                encodings.append(model.encoder(X.to(device)))
            else:
                encodings.append(model.encoder(X.to(device))[0])

    return torch.vstack(encodings)

if __name__ == '__main__':
    import json
    import argparse

    np.float = float

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type = str, default = 'O2VAE_encoding_config.json',
                        help = 'Path to the config file for encoding')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ShapeO2VAE(latent_dim = config['latent_dim'],
                          enc_channels = config['enc_channels'],
                          enc_last_pooling = config['enc_last_pooling'],
                          do_sigmoid= True).to('cuda')

    model.eval()
    model.load_state_dict(torch.load(config['dict_path'], weights_only= False, map_location= torch.device('cuda')))
    encodings = autoencoding(model, config, torch.device('cuda'), True)
    np.save('O2VAE_embeddings.npy', encodings.cpu().numpy())
