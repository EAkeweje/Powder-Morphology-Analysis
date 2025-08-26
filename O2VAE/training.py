import torch
import numpy as np
from tqdm import tqdm

### Helper functions for training, and inference
from tqdm import tqdm
from model_utils import O2Loss


def train_step(
    model, optimizer, criterion, dataloader, device,
    loss_config, accum_steps=1, use_amp=False):
    '''
    One training epoch with optional AMP and gradient accumulation.
    '''
    model.train(True)
    train_loss_ = 0.0
    recon_loss_ = 0.0
    kld_loss_ = 0.0

    if use_amp:
        scaler = torch.amp.GradScaler('cuda')

    optimizer.zero_grad()

    for i, input in enumerate(tqdm(dataloader, ascii=True, desc='training step')):
        input = input.to(device)

        with torch.autocast('cuda', enabled= use_amp):
            if loss_config['model_type'].lower() == 'vae':
                predict, mu, logvar = model(input)
                loss, recon_loss, kld = criterion(predict, input, mu, logvar, loss_config)
                recon_loss_ += recon_loss.item()
                kld_loss_ += kld.item()
            
            elif loss_config['model_type'].lower() == 'vade':
                predict, mu, logvar, z = model(input)
                loss, recon_loss, kld = criterion(predict, input, mu, logvar, z, loss_config)
                recon_loss_ += recon_loss.item()
                kld_loss_ += kld.item()
            
            else:
                predict = model(input)
                loss = criterion(predict, input)

        loss = loss / accum_steps  # Normalize loss

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loss_ += loss.item() * accum_steps  # Unscaled loss

    return train_loss_, recon_loss_, kld_loss_


def valid_step(model, criterion, dataloader, device, loss_config, use_amp = False):
    '''
    One validation epoch with optional AMP.
    '''
    valid_loss_ = 0.0
    model.eval()

    with torch.no_grad():
        for input in tqdm(dataloader, ascii=True, desc='validation step'):
            input = input.to(device)

            with torch.autocast('cuda', enabled = use_amp):
                if loss_config['model_type'].lower() == 'vae':
                    predict, mu, logvar = model(input)
                    loss = criterion(predict, input, mu, logvar, loss_config)[0]

                elif loss_config['model_type'].lower() == 'vade':
                    predict, mu, logvar, z = model(input)
                    loss = criterion(predict, input, mu, logvar, z, loss_config)[0]
                    
                else:
                    predict = model(input)
                    loss = criterion(predict, input)

            valid_loss_ += loss.item()

    return valid_loss_


def training_amp(model, train_loader, val_loader, config, device, use_pretrained = False, use_amp = False):
    '''
    model:: Neural network
    epoch:: 
    optimizer:: optimization algorithm. Default Adam
    learning rate:: 
    dict_path:: path to save best model's state_dict
    criterion:: loss function.
    '''
    if use_amp:
        assert device == torch.device('cuda'), "Training has to be done on CUDA."

    model = model.float()
    
    train_losses, val_losses = [], []
    recon_losses, kld_losses = [], []

    if config['model_type'].lower() not in ['vae', 'vade']:
        criterion = O2Loss(image_shape=(128,128), loss_fun= config['recon_loss_fun'])
    else:
        criterion = model.loss

    if use_pretrained:
        min_valid_loss = valid_step(model, criterion, val_loader, device, loss_config = config, use_amp = use_amp)
    else:
        min_valid_loss = np.inf

    #default weight_decay = 0
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas = (config['beta1'], config['beta2']), weight_decay = config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], momentum= config['SGD_momentum'], weight_decay = config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer ({config['optimizer']})")

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)#old
    assert config['scheduling'] in [ 'step_lr', 'cosine_annealing_lr', False], f"Unknown scheduling type: {config['scheduling']}. Must be one of: 'step_lr', 'cosine_annealing_lr', False."

    if config['scheduling'] == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.9)
    elif config['scheduling'] == 'cosine_annealing_lr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min= 1e-7, T_max=20)
    else:
        scheduler = None


    no_save = 0 #for early stopping

    for e in range(config['epochs']):
        train_loss_, recon_loss_, kld_loss_ = train_step(model, optimizer, criterion, train_loader, device,
                                                         loss_config = config,
                                                         accum_steps = config['accum_steps'],
                                                         use_amp= use_amp)
        valid_loss_ = valid_step(model, criterion, val_loader, device,
                                 loss_config = config,
                                 use_amp = use_amp)

        train_losses.append(train_loss_)
        val_losses.append(valid_loss_)
        recon_losses.append(recon_loss_)
        kld_losses.append(kld_loss_)

        if scheduler:
            scheduler.step()

        if min_valid_loss > valid_loss_:
            min_valid_loss = valid_loss_

            no_save = 0 #reset counter
            # Saving State Dict
            if config['dict_path'] != None:
                # model.to('cpu')
                # model.enc_model.eval()
                torch.save(model.state_dict(), config['dict_path'])

                print(f'Saving Model... Error at Epoch {e+1}: {valid_loss_}')

        else:
            no_save += 1

        # Early stopping
        max_no_save_run = config['max_no_save_run']
        if no_save >= max_no_save_run:
            print(f'Early stopping, no local minimum reach since last {max_no_save_run} epoches')
            break

    return e, min_valid_loss, train_losses, val_losses, recon_losses, kld_losses


if __name__ == "__main__":
    import json
    import argparse
    import torch
    import matplotlib.pyplot as plt

    from model_utils import SO2Loss, O2Loss
    from PowderShapeAutoEncoder import *

    np.float = float

    parser = argparse.ArgumentParser(description='Train O2VAE model')
    parser.add_argument('-tc', '--training_config', type = str, default= 'training_config.json',
                        help= 'Path to triaining config file in JSON format.')
    parser.add_argument('-dc', '--data_config', type = str, default= 'data_config.json',
                        help = 'Path to data config file in JSON format.')
    args = parser.parse_args()

    data_config = json.load(open(args.data_config, 'r'))
    training_config = json.load(open(args.training_config, 'r'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_dataloaders(data_config)

    model = ShapeO2VAE(latent_dim= training_config['latent_dim'],
                enc_channels= training_config['enc_channels'],
                enc_last_pooling= training_config['last_pooling'])
    
    if training_config['use_pretrained']:
        # load pretrained model checkpoint
        print("device:", device )
        model.to(device)
        model_state = torch.load(training_config['dict_path'], map_location= device)
        model.load_state_dict(model_state, strict=True)

    #warm start
    wrms = torch.rand(4,1,128,128)
    wrm = model(wrms.to(device))

    #training
    model = model.to(device)
    e, val_loss, train_losses, val_losses, recon_losses, kld_losses = training_amp(model, train_loader, val_loader, training_config, device,
                                                                                use_pretrained = training_config['use_pretrained'],
                                                                                use_amp = False)
    # loss_plot(train_losses, val_losses)
    print('least val loss:', val_loss)

    plt.plot(np.array(train_losses) / (len(train_loader) * data_config['batch_size']), label = 'train')
    plt.plot(np.array(val_losses) / (len(val_loader) * data_config['batch_size']), label = 'test')
    # plt.yscale('log')
    plt.legend()
