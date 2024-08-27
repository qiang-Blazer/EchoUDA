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
from models.Unet.unet_model_da import UNet_DA_FC
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.utils import set_seed, save_args, segmentation_metrics
from utils.dice_loss import DiceLoss
from itertools import cycle


def train(source_dataloader, target_dataloader, args, epoch):
    source_len = len(source_dataloader)
    target_len = len(target_dataloader)
    max_length = max(source_len, target_len)
    source_iter = cycle(source_dataloader)
    target_iter = cycle(target_dataloader)

    running_loss_seg = 0.0
    running_loss_fc_source = 0.0
    running_loss_fc_target2source = 0.0
    running_metrics = np.zeros(8)
    model.train()  

    for i in range(max_length):
        optimizer.zero_grad()
        images_source, masks_source = next(source_iter)
        images_target, masks_target = next(target_iter)

        images_source = images_source.float().to(device)
        images_target = images_target.float().to(device)
        masks_source = masks_source.float().to(device)
        masks_target = masks_target.float().to(device)
        preds_source, preds_target, loss_fc_source, loss_fc_target2source = model(images_source, images_target, masks_source, epoch)

        loss_seg = criterion_bce(preds_source.squeeze(1), masks_source) + \
                   criterion_dice(F.sigmoid(preds_source), masks_source) 

        #freeze the feature extractor of the model
        if args.freeze:
            if loss_seg<args.loss_seg_of_freeze_threshold and loss_fc_source<0.05:
                for param in model.feature_extractor.parameters():
                    param.requires_grad = False
                loss = loss_fc_target2source
            else:
                for param in model.feature_extractor.parameters():
                    param.requires_grad = True
                loss = loss_seg + loss_fc_source + args.lambda_da*loss_fc_target2source
        loss = loss_seg + loss_fc_source + args.lambda_da*loss_fc_target2source


        running_loss_seg += loss_seg.item()
        running_loss_fc_source += loss_fc_source.item()
        running_loss_fc_target2source += loss_fc_target2source.item()
        running_metrics += np.concatenate((segmentation_metrics(F.sigmoid(preds_source), masks_source), 
                                           segmentation_metrics(F.sigmoid(preds_target), masks_target)))

        loss.backward()
        optimizer.step()
    scheduler.step()

    return running_loss_seg/max_length, running_loss_fc_source/max_length, running_loss_fc_target2source/max_length, running_metrics/max_length


def eval(source_dataloader, target_dataloader, epoch):
    source_len = len(source_dataloader)
    target_len = len(target_dataloader)
    max_length = max(source_len, target_len)
    source_iter = cycle(source_dataloader)
    target_iter = cycle(target_dataloader)

    running_loss_seg = 0.0
    running_loss_fc_source = 0.0
    running_loss_fc_target2source = 0.0
    running_metrics = np.zeros(8)
    
    model.eval()  
    with torch.no_grad():
        for i in range(max_length):
            optimizer.zero_grad()
            images_source, masks_source = next(source_iter)
            images_target, masks_target = next(target_iter)

            images_source = images_source.float().to(device)
            images_target = images_target.float().to(device)
            masks_source = masks_source.float().to(device)
            masks_target = masks_target.float().to(device)
            preds_source, preds_target, loss_fc_source, loss_fc_target2source = model(images_source, images_target, masks_source, epoch)

            loss_seg = criterion_bce(preds_source.squeeze(1), masks_source) + \
                       criterion_dice(F.sigmoid(preds_source), masks_source) 

            running_loss_seg += loss_seg.item()
            running_loss_fc_source += loss_fc_source.item()
            running_loss_fc_target2source += loss_fc_target2source.item()
            running_metrics += np.concatenate((segmentation_metrics(F.sigmoid(preds_source), masks_source), 
                                               segmentation_metrics(F.sigmoid(preds_target), masks_target)))

    return running_loss_seg/max_length, running_loss_fc_source/max_length, running_loss_fc_target2source/max_length, running_metrics/max_length


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
    parser.add_argument('--target_dataset', type=str, default="CardiacUDA", choices=['EchoNet_Dynamic', 'CardiacUDA', 'CAMUS', 'EchoNet_Pediatric', 'HMC_QU'], help='Target dataset')
    parser.add_argument('--view', type=str, default="4CH", help='Echo view')
    parser.add_argument('--position', type=str, default="LV", help='The part of the echo to segmentation')  
    parser.add_argument('--result_dir', type=str, default="runs/0", help='Directory to save results')
    parser.add_argument('--pretrained_dir', type=str, default=None, help='Directory to load pretrained checkpoints')
    parser.add_argument('--lambda_da', type=float, default=1.0, help='Weight of the domain adaptation loss in the total loss')
    parser.add_argument('--freeze', type=bool, default=False, help='Whether to freeze the feature extractor of the model')
    parser.add_argument('--loss_seg_of_freeze_threshold', type=float, default=0.10, help='The threshold of loss_seg to freeze the feature extractor of the model, only used when freeze=True')
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
    train_dataset_target = getattr(Dataset_image_mask, args.target_dataset)(train=True, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    val_dataset_target = getattr(Dataset_image_mask, args.target_dataset)(train=False, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    train_dataset_source = getattr(Dataset_image_mask, args.source_dataset)(train=True, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    val_dataset_source = getattr(Dataset_image_mask, args.source_dataset)(train=False, view=args.view, position=args.position, r_size=args.r_size, random_seed=args.seed)
    train_dataloader_target = DataLoader(train_dataset_target, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_mem, num_workers=args.num_workers)
    val_dataloader_target = DataLoader(val_dataset_target, batch_size=args.batch_size, shuffle=False,pin_memory=args.pin_mem, num_workers=args.num_workers)
    train_dataloader_source = DataLoader(train_dataset_source, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_mem, num_workers=args.num_workers)
    val_dataloader_source = DataLoader(val_dataset_source, batch_size=args.batch_size, shuffle=False,pin_memory=args.pin_mem, num_workers=args.num_workers)
    
    #load the model
    model = UNet_DA_FC(n_channels=1, n_classes=1)
    model.to(device)
    if args.pretrained_dir is not None:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "model.pth")))

    #criterion 
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce.to(device)
    criterion_dice.to(device)
    criterion_ce.to(device)
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
        loss_seg,loss_fc_source,loss_fc_target2source,metrics = train(train_dataloader_source, train_dataloader_target, args, epoch)
        log = f'Epoch: {epoch:03d}, [Train] Seg Loss: {loss_seg:.4f}, FC S Loss: {loss_fc_source:.4f}, T2S Loss: {loss_fc_target2source:.4f}, S Dice: {metrics[0]:.4f}, S IOU: {metrics[1]:.4f}, S ASSD: {metrics[2]:.4f}, S HD: {metrics[3]:.4f}, T Dice: {metrics[4]:.4f}, T IOU: {metrics[5]:.4f}, T ASSD: {metrics[6]:.4f}, T HD: {metrics[7]:.4f}'
        print(log)
        training_logs += log+"\n" 
        loss_seg,loss_fc_source,loss_fc_target2source,metrics = eval(val_dataloader_source, val_dataloader_target, epoch)
        log = f'Epoch: {epoch:03d}, [Val] Seg Loss: {loss_seg:.4f}, FC S Loss: {loss_fc_source:.4f}, T2S Loss: {loss_fc_target2source:.4f}, S Dice: {metrics[0]:.4f}, S IOU: {metrics[1]:.4f}, S ASSD: {metrics[2]:.4f}, S HD: {metrics[3]:.4f}, T Dice: {metrics[4]:.4f}, T IOU: {metrics[5]:.4f}, T ASSD: {metrics[6]:.4f}, T HD: {metrics[7]:.4f}'
        print(log)
        training_logs += log+"\n" 
        if metrics[4] > best_metric:
            best_metric = metrics[4]  
            torch.save(model.state_dict(), os.path.join(result_dir,"model.pth"))

    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()







