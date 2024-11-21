import numpy as np
import torch
from torch import Tensor
import cv2
import os
import argparse
import logging
import colorlog
from argparse import  Namespace
from torch.nn  import  functional as F
from skimage import  morphology
import  copy



def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def  calc_predict_information(predict):
    assert type(predict) == np.ndarray
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(predict, connectivity=8)
    # retval连通域个数   #labels连通图  #stats连通域大小和范围  #连通域质心
    retval = retval - 1
    ind = stats[:, 4].argsort()
    stats = stats[ind][:-1]
    centroids=centroids[ind][:-1]
    h,w=predict.shape
    predict=predict//255
    predict=predict[:,:,None]   #维度扩充 [h,w,1]  0-1  ndarray
    ###################################################################

    ######preporcessing##########################
    final_index = []
    for i in range(retval):
        x1, y1, x2, y2 = stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            final_index.append(i)
        else:
            pass
            # print(f"position:{x1}, {y1}, {x2}, {y2}")
    #################################################
    retval = len(final_index)
    stats = stats[final_index]
    centroids = centroids[final_index]
    #################################################
    if retval == 0:
        # print(f"the number of retval is {retval} ")
        retval+=1
        stats=np.array([[0,0,w,h]])
        centroids=np.array([[w//2,h//2]])

    return retval,labels,stats,centroids,predict




def  calc_information(mask_rpath):
    mask = cv2.imread(mask_rpath, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # retval连通域个数   #labels连通图  #stats连通域大小和范围  #连通域质心

    #####################去掉最大外接框##########################
    retval = retval - 1
    index = stats[:, 4].argsort()        #sort from small to large  according to the areas and return index
    stats = stats[index][:-1]
    centroids=centroids[index][:-1]
    ##########################################################

    ######preporcessing##########################
    final_index=[]
    for i in range(retval):
        x1, y1, x2, y2 = stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            final_index.append(i)
   #################################################
    retval=len(final_index)
    stats=stats[final_index]
    centroids = centroids[final_index]
    #################################################
    mask=mask//255
    mask=mask[:,:,None]    # [h,w,1]   0-1  ndarray
    return retval,labels,stats,centroids,mask

def load_model(args):
    if args.model_name=="SAM":
        from SAM.segment_anything import  sam_model_registry,SamPredictor
        sam = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth").cuda()
        predictor = SamPredictor(sam)
    elif args.model_name=="SAM-B":
        from SAM.segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_vit_b_01ec64.pth").cuda()
        predictor = SamPredictor(sam)
    elif args.model_name=="MedSAM_bbox":
        from SAM.segment_anything import sam_model_registry,SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/medsam_vit_b.pth").cuda()
        predictor = SamPredictor(sam)
    elif args.model_name=="MedSAM_point":
        from SAM.segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/medsam_point_prompt_flare22.pth").cuda()
        predictor = SamPredictor(sam)
    else:
        raise RuntimeError("model_name must be  SAM or MedSAM")

    return predictor


def print_args(args:argparse.Namespace,logger):
    for k,v in  args.__dict__.items():
        logger.info(f"{k}:{v}")




def get_logger(log_path):
    # 第一步：创建日志器
    logger = logging.getLogger("tech_stu")
    logger.setLevel(logging.INFO)
    # 第二步：定义处理器。控制台和文本输出两种方式
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    # 第三步：设置的不同的输入格式

    console_fmt ='%(log_color)s %(message)s'
    file_fmt = '%(levelname)s    %(asctime)s    %(message)s'
    # 第三步：格式
    # fmt1 = logging.Formatter(fmt=console_fmt)
    fmt1 = colorlog.ColoredFormatter(fmt=console_fmt,log_colors={"INFO": "green", "WARNING": "yellow", "ERROR": "red"})
    fmt2 = logging.Formatter(fmt=file_fmt)
    # 第四步:把格式传给处理器
    console_handler.setFormatter(fmt1)
    file_handler.setFormatter(fmt2)
    # 第五步:把处理器传个日志器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger





def  binary(rrr):
    raw=copy.deepcopy(rrr)
    raw[raw>0.5]=1
    raw[raw<=0.5]=0
    return raw


def calc_params():
    from SAM.segment_anything import sam_model_registry, SamPredictor
    net = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_vit_b_01ec64.pth").cuda()
    n_parameters = sum(p.numel() for p in net.parameters() )
    print(n_parameters)


