from .unet_parts import *
import torch
import math


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_with_feature(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_with_feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, x


   

class UNet_DA_FC_pretrain(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_feature_channels=1):
        super(UNet_DA_FC_pretrain, self).__init__()
        self.feature_extractor = UNet_with_feature(n_channels, n_feature_channels)
        self.segmentation = UNet(n_feature_channels, n_classes)

    def forward(self, x, y):
        y_pred, x_feature = self.feature_extractor(x)
        N,C,H,W = x_feature.shape
        x_feature = x_feature.permute(0,2,3,1).reshape(N*H*W,C)
        y = y.reshape(N*H*W)

        x_feature_fg_mean = x_feature[y==1].mean(dim=0)
        x_feature_bg_mean = x_feature[y==0].mean(dim=0)

        loss_x_fc = (torch.matmul(x_feature_fg_mean,x_feature_bg_mean.T))**2

        y_pred = self.segmentation(y_pred)

        return y_pred, loss_x_fc


class UNet_DA_FC(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_feature_channels=1):
        super(UNet_DA_FC, self).__init__()
        self.feature_extractor = UNet_with_feature(n_channels, n_feature_channels)
        self.segmentation = UNet(n_feature_channels, n_classes)

    def forward(self, x1, x2, y1, epoch):
        y1_pred, x1_feature = self.feature_extractor(x1)
        y2_pred, x2_feature = self.feature_extractor(x2)
        N1,C,H,W = x1_feature.shape
        N2 = x2_feature.shape[0]
        x1_feature = x1_feature.permute(0,2,3,1).reshape(N1*H*W,C)
        x2_feature = x2_feature.permute(0,2,3,1).reshape(N2*H*W,C)
        y1 = y1.reshape(N1*H*W)

        x1_feature_fg_mean = x1_feature[y1==1].mean(dim=0)
        x1_feature_bg_mean = x1_feature[y1==0].mean(dim=0)

        loss_x1_fc = (torch.matmul(x1_feature_fg_mean,x1_feature_bg_mean.T))**2

        y1_pred = self.segmentation(y1_pred)
        y2_pred = self.segmentation(y2_pred)
        y2_pred_prob = F.sigmoid(y2_pred.reshape(N2*H*W))    
        if epoch%4 == 0:
            delta = 0.01 * math.exp(0.01*epoch)
            thres1 = torch.quantile(y2_pred_prob, 0.95-delta)
            thres2 = torch.quantile(y2_pred_prob, 0.05+delta)
            x2_feature_pred_fg = x2_feature[y2_pred_prob>thres1]
            x2_feature_pred_bg = x2_feature[y2_pred_prob<thres2]
            #print(thres1, thres2)

            loss_x2_pred_fg_to_x1_fg = 1-F.cosine_similarity(x2_feature_pred_fg.mean(dim=0).unsqueeze(0), x1_feature_fg_mean)
            loss_x2_pred_bg_to_x1_bg = 1-F.cosine_similarity(x2_feature_pred_bg.mean(dim=0).unsqueeze(0), x1_feature_bg_mean)
            loss_x2_fc = (torch.matmul(x2_feature_pred_fg.mean(dim=0),x2_feature_pred_bg.mean(dim=0).T))**2
            
            loss_x2 = loss_x2_pred_fg_to_x1_fg + loss_x2_pred_bg_to_x1_bg + loss_x2_fc 

        else:
            x2_feature_to_x1_fg_sim = F.cosine_similarity(x2_feature, x1_feature_fg_mean)
            x2_feature_to_x1_bg_sim = F.cosine_similarity(x2_feature, x1_feature_bg_mean)
            x2_feature_to_x1_sim = torch.softmax(torch.cat((x2_feature_to_x1_fg_sim.unsqueeze(0), x2_feature_to_x1_bg_sim.unsqueeze(0))), dim=0)
            y2_pred_prob = torch.cat((y2_pred_prob.unsqueeze(0), (1-y2_pred_prob).unsqueeze(0))) 
            loss_x2_feature_to_pred_consistency = -(torch.log(x2_feature_to_x1_sim)*y2_pred_prob).sum(dim=0).mean()

            loss_x2 = loss_x2_feature_to_pred_consistency

        return y1_pred, y2_pred, loss_x1_fc, loss_x2


