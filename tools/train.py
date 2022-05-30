import pandas as pd
import numpy as np
import yaml
from dataset import CLIPDataset, get_transforms
import torch
from tqdm import tqdm
from utils import AvgMeter, get_lr, load_config
from transformers import DistilBertTokenizer
from models import CLIPModel
import itertools
import argparse


def label_preprocessing(config):
    df = pd.read_csv(config['global']['label_path'], delimiter="|")
    df.columns = ['image', 'caption_number', 'caption']
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
    ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
    df['id'] = ids
    df.to_csv(config['global']['captions_path'], index=False)



def make_train_valid_dfs(config):
    dataframe = pd.read_csv(config['global']['captions_path'])
    max_id = dataframe["id"].max() + 1 if not config['global']['debug'] else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode, config):
    transforms = get_transforms(mode="train", config=config)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['global']['batch_size'],
        num_workers=config['global']['num_workers'],
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, device, train_loader, optimizer, lr_scheduler, step, config):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, device, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(config['text_encoder']['text_tokenizer'])
    train_loader = build_loaders(train_df, device, tokenizer, mode="train", config=config)
    valid_loader = build_loaders(valid_df, device, tokenizer, mode="valid", config=config)

    model = CLIPModel(config).to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": config['image_encoder']['image_encoder_lr']},
        {"params": model.text_encoder.parameters(), "lr": config['text_encoder']['text_encoder_lr']},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": config['global']['head_lr'], "weight_decay": config['global']['weight_decay']}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config['global']['patience'], factor=config['factor']
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(config['global']['epochs']):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f"epoch_{epoch}_loss_{train_loss}.pt")

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f"best_loss_{best_loss}.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

