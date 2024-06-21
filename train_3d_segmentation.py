# %%
import os
import time
import pathlib
import torch
from typing import NoReturn

from losses import Dice_and_FocalLoss
from model import FastSmoothSENormDeepUNet_supervision_skip_no_drop as se_model

from param_loader import Params
from torch.utils.data import DataLoader
import pandas as pd
import metrics
import dataset_hn2
import augmentation as transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, loss_fn, train_dl, metric, scheduler=False,
          device=DEVICE):
    model.train()
    epoch_loss = 0
    epoch_metric = 0
    batch_nr = 0
    for batch in train_dl:
        load_time_start = time.time()
        optimizer.zero_grad()
        input = batch['input']
        target = batch['target']
        input = input.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        prediction = model(input)
        loss = loss_fn(prediction, target)
        metric_score = metric(prediction.detach(), target.detach())

        epoch_loss += loss.item()
        epoch_metric += metric_score.item()
        del input, target, prediction, metric_score

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        del loss

        batch_nr += 1

    epoch_loss /= batch_nr
    epoch_metric /= batch_nr

    return epoch_loss, epoch_metric


def evaluate(model, loss_fn, val_dl, metric, device=DEVICE):
    model.eval()
    epoch_loss = 0
    epoch_metric = 0
    batch_nr = 0
    with torch.no_grad():
        for batch in val_dl:
            input = batch['input']
            target = batch['target']
            input = input.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            prediction = model(input)
            loss = loss_fn(prediction, target)
            metric_score = metric(prediction.detach(), target.detach())

            epoch_loss += loss.item()
            epoch_metric += metric_score.item()

            del input, target, prediction, metric_score
            del loss

            batch_nr += 1

        epoch_loss /= batch_nr
        epoch_metric /= batch_nr
    return epoch_loss, epoch_metric


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def train_main(data_path: str,
               model_path: str,
               training_cycle: int,
               save_model_path: str,
               logs_path: str) -> NoReturn:

    print(data_path)
    param_path = pathlib.Path('/home/params.json')
    params = Params(param_path)

    base_name = os.path.basename(data_path)
    save_model_path = os.path.join(save_model_path, base_name)
    logs_path = os.path.join(logs_path, base_name)

    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    train_transforms = transforms.Compose([
                        transforms.RandomRotation(p=0.5, angle_range=[params.dict['rotation_l'], params.dict['rotation_r']]),
                        transforms.Mirroring(p=0.5),
                        transforms.ToTensor()])

    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    epoch_list = []
    epoch_list_val = []
    train_loss_list = []
    validation_loss_list = []
    train_dice_list = []
    validation_dice_list = []
    highest_val_hci = 0

    train_path = os.path.join(data_path, 'Train')
    validation_path = os.path.join(data_path, 'Validation')
    train_set = dataset_hn2.HNDataset(train_path,
                                      params.dict['shift'],
                                      transform=train_transforms)
    val_set = dataset_hn2.HNDatasetVal(validation_path,
                                       transform=val_transforms)

    train_dl = DataLoader(train_set, batch_size=params.dict['batch_size'],
                          shuffle=True)
    val_dl = DataLoader(val_set, batch_size=params.dict['batch_size'],
                        shuffle=False)

    model = se_model(in_channels=1, n_cls=2, n_filters=16).to(device=DEVICE)

    loss_fn = Dice_and_FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params.dict['learning_rate'],
                                 betas=(0.9, 0.99),
                                 weight_decay=0.0001)

    metric = metrics.dice
    # T_0:25 -> 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10,
                                                                     eta_min=1e-6)

    # If the provided training_cycle > 0, load model from "model_path".
    if training_cycle != 0 and model_path is not None:
        # checkpoint = torch.load(model_path)
        print(f'Loading model from: {model_path}')
        model.load_state_dict(torch.load(model_path))
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    else:
        print(f'Starting training from scratch')

    step = 0
    for epoch in range(params.dict["num_epochs"]):
        epoch_loss, epoch_metric = train(model,
                                         optimizer,
                                         loss_fn,
                                         train_dl,
                                         metric,
                                         scheduler)

        # avg_epoch_loss += loss_value
        print(f'batch: {epoch} batch loss: {epoch_loss:.3f} \tdice: {epoch_metric:.3f}')
        print()
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss)
        train_dice_list.append(epoch_metric)
        # save model
        if epoch % params.dict["save_steps"] == 0:
            model_name = 'model_weights' + str(epoch) + '.pt'
            save_path = pathlib.Path(save_model_path)
            if not os.path.exists(save_path):
                save_path.mkdir(parents=True)
            torch.save(model.state_dict(),
                       os.path.join(save_path, model_name))
            full_model_name = 'model_state' + str(epoch) + '.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'scheduler': scheduler}, os.path.join(save_model_path,
                                                              full_model_name))

        if epoch % params.dict["eval_steps"] == 0:
            val_loss, val_dsc = evaluate(model, loss_fn, val_dl, metric,
                                         device=DEVICE)
            print(f'Val batch: {epoch} batch loss: {val_loss:.3f} \tdice: {val_dsc:.3f}')
            print()
            epoch_list_val.append(epoch)
            validation_loss_list.append(val_loss)
            validation_dice_list.append(val_dsc)
            if val_dsc > highest_val_hci:
                highest_val_hci = val_dsc
                model_name = 'highest_val' + '.pt'
                save_path = pathlib.Path(save_model_path)
                if not os.path.exists(save_path):
                    save_path.mkdir(parents=True)
                torch.save(model.state_dict(),
                           os.path.join(save_path, model_name))
                full_model_name = 'highsted_model_state' + str(epoch) + '.pt'
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'scheduler': scheduler},
                           os.path.join(save_model_path, full_model_name))
        step += 1
    logs_path = pathlib.Path(logs_path)
    df = pd.DataFrame({'epoch': epoch_list,
                       'train_loss': train_loss_list,
                       'train_dice': train_dice_list})
    df.to_csv(logs_path / 'train.csv', index=False)
    df2 = pd.DataFrame({'epoch': epoch_list_val,
                        'val_loss': validation_loss_list,
                        'val_dice': validation_dice_list})
    df2.to_csv(logs_path / 'validation.csv', index=False)

# %%
