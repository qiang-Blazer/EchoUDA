import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from scipy.spatial.distance import directed_hausdorff


def set_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_runs_dir(result_name):
    run_times = len(os.listdir(result_name))+1
    result_dir = os.path.join(result_name,str(run_times))
    os.makedirs(result_dir,exist_ok=True)

    return result_dir


def save_args(args, result_dir):  
    args_dict = args.__dict__
    with open(os.path.join(result_dir,'args.txt'), 'w') as f:
        for key, value in args_dict.items():
            f.writelines(key + ' : ' + str(value) + '\n')
        f.close()


def get_auc_and_threshold(label,pred):
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    index = np.argmax(tpr-fpr)
    optimal_threshold = thresholds[index]
    
    return roc_auc,optimal_threshold,fpr,tpr


def get_roc_curve(ax, fpr, tpr, roc_auc, result_dir, metric='auc'):    
    ax.clear()
    ax.plot(fpr, tpr, color='#CC0033', lw=4, label=f'ROC curve (AUC = {roc_auc:.4f})') 
    ax.plot([0,1], [0,1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.05)
    ax.tick_params(labelsize=18)
    ax.set_xlabel('False Positive Rate',fontdict={'size':20},labelpad=12)
    ax.set_ylabel('True Positive Rate',fontdict={'size':20},labelpad=12)
    ax.set_title('ROC curve',fontdict={'size':22})
    ax.legend(loc="lower right",fontsize=20)
    plt.savefig(os.path.join(result_dir,f"roc-curve_{metric}.png"))


def dice_score_and_iou(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    return dice, iou


def assd_and_hausdorff_distance(pred, true):
    pred_surface = np.argwhere(pred)
    true_surface = np.argwhere(true)
    if len(pred_surface) == 0 or len(true_surface) == 0:
        return 100,100
    d1 = directed_hausdorff(pred_surface, true_surface)[0]
    d2 = directed_hausdorff(true_surface, pred_surface)[0]
    assd = (d1 + d2) / 2
    hd = max(d1, d2)

    return assd, hd


def segmentation_metrics(preds, targets):
    # Ensure preds and targets are binary (0, 1)
    preds = (preds > 0.5).float().cpu().numpy().squeeze()
    targets = targets.float().cpu().numpy().squeeze()

    dice_scores = []
    iou_scores = []
    assd_scores = []
    hd_scores = []

    for i in range(preds.shape[0]):
        pred = preds[i]
        target = targets[i]

        dice, iou = dice_score_and_iou(pred, target)
        assd, hd = assd_and_hausdorff_distance(pred, target)
        
        dice_scores.append(dice)
        iou_scores.append(iou)        
        assd_scores.append(assd)
        hd_scores.append(hd)

    return np.mean(dice_scores), np.mean(iou_scores), np.mean(assd_scores), np.mean(hd_scores)


def tensor2image(tensor):
    image = (tensor * 0.25 + 0.5).clamp(0, 1).squeeze(0).detach().cpu().float().numpy()*255.0
    return image.astype(np.uint8)


def draw(images):
    for image_name, tensor in images.items():
        image = Image.fromarray(tensor2image(tensor))
        image.save(os.path.join('runs', image_name+'.png'))

# used in advent
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30))