
import os
import cv2
import numpy as np
import torch
from monai.metrics import  compute_hausdorff_distance

from argparse import  Namespace


def  calc_dice(pred,gt):
    assert pred.max()<=1 and gt.max()<=1
    pred=np.reshape(pred,newshape=(1,-1))
    gt=np.reshape(gt,newshape=(1,-1))
    smooth = 1
    intersection = (pred*gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def calc_hf(pred,gt):
    h,w=pred.shape
    pred=torch.from_numpy(pred)
    gt=torch.from_numpy(gt)
    A=pred[None,:,:]
    B=gt[None,:,:]


    A=A.unsqueeze(0)
    B=B.unsqueeze(0)
    hf=compute_hausdorff_distance(A,B)
    thres=(h**2+w**2)**0.5
    if hf>thres:
        hf=torch.tensor(thres)

    return hf.item()



def dice_and_hf(args,logger,directory = "Polyp",datasets = [ "CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"]):
    for dataset in datasets:
        mask_path = f"./TestDatasets/{directory}/{dataset}/masks"
        # mask_path = f"./Fluid/fluidchallenge_test/{dataset}/masks"
        predict_path = f"./TestDatasets/{directory}/{dataset}/predicts_SAM_H_point"
        if  args.result_path!="":
            predict_path=f"{args.result_path}/{dataset}"
        sample_list=sorted(os.listdir(mask_path))
        temp_dice=[]
        temp_hf=[]

        for index,sample_name in enumerate(sample_list):
            predict=cv2.imread(os.path.join(predict_path,sample_name),0)
            mask=cv2.imread((os.path.join(mask_path,sample_name)),0)
            _,predict=cv2.threshold(predict,127,255,cv2.THRESH_BINARY)
            _,mask=cv2.threshold(mask,127,255, cv2.THRESH_BINARY)
            predict=predict/255
            mask=mask/255
            dice=calc_dice(predict,mask)
            hf=calc_hf(predict,mask)

            temp_dice.append(dice)
            temp_hf.append(hf)



        mdice=round(sum(temp_dice)/len(temp_dice),4)
        mdf=round(sum(temp_hf)/len(temp_hf),2)

        logger.info(f"{dataset}_dice_{mdice}")
        logger.info(f"{dataset}_hf_{mdf}")

if  __name__=="__main__":
    args=Namespace()
    args.result_path="./result/exp_2024_02_26_23_02_50"
    # directory = "BUSI"
    # datasets = ["benign","malignant"]
    directory = "fluidchallenge"
    datasets = ["cirrus"]  # "spectralis","topcon"
    # directory = "QU"
    # datasets = ["benign", "malignant"]
    dice_and_hf(args=args,logger=None,directory=directory,datasets=datasets)
