import os
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import Dataset_image_mask
from models.Unet.unet_model_da import UNet_DA_FC_pretrain
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.utils import set_seed, save_args, segmentation_metrics
from utils.dice_loss import DiceLoss
from itertools import cycle


def train(source_dataloader):
    source_len = len(source_dataloader)
    running_loss_seg = 0.0
    running_loss_fc = 0.0
    running_metrics = np.zeros(4)
    model.train()  

    for images_source, masks_source in source_dataloader:
        optimizer.zero_grad()
        images_source = images_source.float().to(device)
        masks_source = masks_source.float().to(device)
        preds_source, loss_fc = model(images_source, masks_source)

        loss_seg = criterion_bce(preds_source.squeeze(1), masks_source) + \
                   criterion_dice(F.sigmoid(preds_source), masks_source) 
        loss = loss_seg + loss_fc
        running_loss_seg += loss_seg.item()
        running_loss_fc += loss_fc.item()
        running_metrics += segmentation_metrics(F.sigmoid(preds_source), masks_source)          

        loss.backward()
        optimizer.step()
    scheduler.step()

    return running_loss_seg/source_len, running_loss_fc/source_len, running_metrics/source_len


def eval(source_dataloader):
    source_len = len(source_dataloader)
    running_loss_seg = 0.0
    running_loss_fc = 0.0
    running_metrics = np.zeros(4)
    
    model.eval()  
    with torch.no_grad():
        for images_source, masks_source in source_dataloader:

            images_source = images_source.float().to(device)
            masks_source = masks_source.float().to(device)
            preds_source, loss_fc = model(images_source, masks_source)

            loss_seg = criterion_bce(preds_source.squeeze(1), masks_source) + \
                    criterion_dice(F.sigmoid(preds_source), masks_source) 
            running_loss_seg += loss_seg.item()
            running_loss_fc += loss_fc.item()
            running_metrics += segmentation_metrics(F.sigmoid(preds_source), masks_source)  

    return running_loss_seg/source_len, running_loss_fc/source_len, running_metrics/source_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0', help='Cuda id to use')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs') 
    parser.add_argument('--batch_size', type=int, default=2, help='Mini-batch size in training')
    parser.add_argument('--pin_mem', type=bool, default=True, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU, but need more memory')
    parser.add_argument('--num_workers', type=int, default=0, help='Thread numbers used in DataLoader')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--r_size', type=int, default=112, help='Resized frame height and width')
    parser.add_argument('--lr', type=float, default=0.0001, help='Starting learning rate')
    parser.add_argument('--scheduler', type=str, default="cosineannealinglr", help='Scheduler used in training')
    parser.add_argument('--source_dataset', type=str, default="EchoNet_Dynamic", choices=['EchoNet_Dynamic', 'CardiacUDA', 'CAMUS', 'EchoNet_Pediatric', 'HMC_QU'], help='Source dataset')
    parser.add_argument('--view', type=str, default="4CH", help='Echo view')
    parser.add_argument('--position', type=str, default="LV", help='The part of the echo to segmentation')  
    parser.add_argument('--result_dir', type=str, default="runs/0", help='Directory to save results')
    parser.add_argument('--pretrained_dir', type=str, default=None, help='Directory to load pretrained checkpoints')
    args = parser.parse_args()
    args_dict = args.__dict__
    print("------Arguments------")
    for key, value in args_dict.items():
        print(key + ' : ' + str(value))
    set_seed(args.seed)

    #make dir to save results
    result_dir = args.result_dir
    os.makedirs(result_dir,exist_ok=True)
    save_args(args, result_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load the data
    train_dataset_source = getattr(Dataset_image_mask, args.source_dataset)(train=True, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    val_dataset_source = getattr(Dataset_image_mask, args.source_dataset)(train=False, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    train_dataloader_source = DataLoader(train_dataset_source, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_mem, num_workers=args.num_workers)
    val_dataloader_source = DataLoader(val_dataset_source, batch_size=args.batch_size, shuffle=False,pin_memory=args.pin_mem, num_workers=args.num_workers)
    
    #load the model
    model = UNet_DA_FC_pretrain(n_channels=1, n_classes=1)
    model.to(device)
    if args.pretrained_dir is not None:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "model.pth")))

    #criterion 
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_bce.to(device)
    criterion_dice.to(device)
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    #scheduler
    if args.scheduler.lower() == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=20) #drop to 0 at 20/60/100/140, restart at 40/80/120/160 
    elif args.scheduler.lower() == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0) #drop to 0 and restart at 20/60/140/300 
    else:
        raise ValueError(f"The argument --scheduler can not be {args.scheduler}")

    #train the model
    training_logs = ""
    print('------Trainging------')
    torch.autograd.set_detect_anomaly(True)
    best_metric = 0
    for epoch in range(1,args.epoch+1):
        loss_seg,loss_fc,metrics = train(train_dataloader_source)
        log = f'Epoch: {epoch:03d}, [Train] Seg Loss: {loss_seg:.4f}, FC Loss: {loss_fc:.4f}, S Dice: {metrics[0]:.4f}, S IOU: {metrics[1]:.4f}, S ASSD: {metrics[2]:.4f}, S HD: {metrics[3]:.4f}'
        print(log)
        training_logs += log+"\n" 
        loss_seg,loss_fc,metrics = eval(val_dataloader_source)
        log = f'Epoch: {epoch:03d}, [Val] Seg Loss: {loss_seg:.4f}, FC Loss: {loss_fc:.4f}, S Dice: {metrics[0]:.4f}, S IOU: {metrics[1]:.4f}, S ASSD: {metrics[2]:.4f}, S HD: {metrics[3]:.4f}'
        print(log)
        training_logs += log+"\n" 
        if metrics[0] > best_metric:
            best_metric = metrics[0]  
            torch.save(model.state_dict(), os.path.join(result_dir,"model.pth"))

    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()







