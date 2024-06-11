import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import warnings
from pathlib import Path
from metric import calc_dice
from tqdm import tqdm
from torchvision  import  transforms
from Nets.Unet.unet_model import UNet
from tool import *
from metric import dice_and_hf
import random
warnings.filterwarnings("ignore")
import pandas as pd

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()




def predict_with_prompt(args, logger, directory=None,parts=None):
    predictor = load_model(args)
    args.input_channels=4  if args.prompt=="bbox" and args.four_channel else 3
    print(f"channels:{args.input_channels}")
    net = UNet(n_channels=args.input_channels, n_classes=args.nclasses, bilinear=True)
    net.train().cuda()

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0005)
    # logger.info(optimizer)

    branch_image = []
    branch_mask = []

    if args.calc_raw_dice:
        raw_dices=[]

    if args.dice_percent>0:
        gt_percent_queue=[]
    alphas=[]
    sub_sample_names=[]
    sub_parts=[]
    mean_alphas=[]
    last={"part":[],"sample_names":[],"final_dice":[]}


    for idx, part in enumerate(parts):
        image_path = f"./TestDatasets/{directory}/{part}/images"
        mask_path = f"./TestDatasets/{directory}/{part}/masks"

        predict_path = f"{args.result_path}/{part}"
        Path(predict_path).mkdir(parents=True, exist_ok=True)
        view_path = f"{args.result_path}/view/{part}"
        Path(view_path).mkdir(parents=True, exist_ok=True)





        sample_list = sorted(os.listdir(image_path))
        sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

        if  args.dice_percent>0:
            count=0

        for index, sample_name in tqdm(enumerate(sample_list), desc=f"{part}"):
            flag= index%args.interval==0
            retval, _, stats,centroids, mask = calc_bbox(os.path.join(mask_path, sample_name))   #  mask [h,w,1]  0-1
            image = cv2.imread(os.path.join(image_path, sample_name))  # [h,w,c]   [0-255]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h, image_w = image.shape[:2]
            predictor.set_image(image)
            pre_merge = np.zeros(image.shape, dtype=np.float32)  # [h,w,c]  0-1  ndarray
            if args.dice_percent > 0 or args.calc_raw_dice :
                raw_merge = np.zeros((image.shape[0],image.shape[1],1), dtype=np.float32)  # [h,w,c]  0-1  ndarray
            crop_image_queue = []  # [h,w,c]    0-255   [nadarray]
            crop_predict_queue = []  # [h,w,1]    0-1     [nadarray]
            crop_predict_sigmoid_queue=[]  #[h,w,1]  0~1  [ndarray]
            crop_mask_queue = []  # [h,w,1]    0-1     [nadarray]
            unet_predict_queue = []  # [h,w,1]  0-1   ndarray
            crop_bbox_queue=[]
            crop_merge_queue = []
            crop_predict_logist_queue=[]
            unet_logist_queue=[]

            #核心
            for i in range(retval):
                if args.prompt=="bbox":
                    x1, y1, x2, y2 = stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]
                    if x2 - x1 >= 16 and y2 - y1 >= 16:
                        input_bbox = np.array([[x1, y1, x2, y2]])
                        pred_raw, _, _ = predictor.predict(box=input_bbox, multimask_output=False, return_logits=True)
                        pred = torch.from_numpy(pred_raw).sigmoid().permute(1, 2, 0).numpy().astype(np.float32)  # [0~1]  ndarray
                        if args.dice_percent > 0 or args.calc_raw_dice :
                            raw_merge+= pred
                        crop_image_queue.append(image[y1:y2, x1:x2])
                        crop_mask_queue.append(mask[y1:y2, x1:x2])
                        crop_bbox_queue.append([x1, y1, x2, y2])

                        crop_predict_sigmoid_queue.append(pred[y1:y2, x1:x2])  # [h,w,
                        crop_predict_logist_queue.append(torch.from_numpy(pred_raw).permute(1, 2, 0).numpy().astype(np.float32)[y1:y2, x1:x2])

                        pred_mask = (pred > 0.5).astype(np.float32)
                        crop_predict_queue.append(pred_mask[y1:y2, x1:x2])
                else:
                    input_point = np.array([centroids[i]])
                    input_label = np.array([1])
                    pred_raw, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label,multimask_output=False, return_logits=True)
                    pred = torch.from_numpy(pred_raw).sigmoid().permute(1, 2, 0).numpy().astype(np.float32)  # [0~1]  ndarray
                    if args.dice_percent > 0 or args.calc_raw_dice :
                        raw_merge+=pred
                    temp_full = morphology.convex_hull_image(binary(pred[:, :, 0]))
                    temp_full = np.array(temp_full, dtype=np.uint8) * 255
                    _, _, pred_stats, _, _ = calc_predict_bbox(temp_full)
                    if len(pred_stats) != 0:
                        x1, y1, x2, y2 = pred_stats[0][0], pred_stats[0][1], pred_stats[0][0] + pred_stats[0][2], pred_stats[0][1] + pred_stats[0][3]
                        # x1 = int(0.9 * x1)
                        # x2 = min(image_w, int(1.1 * x2))
                        # y1 = int(0.9 * y1)
                        # y2 = min(image_h, int(1.1 * y2))


                        if x2 - x1 >= 16 and y2 - y1 >= 16:
                            x1 = max(0, int(x1 - (x2 - x1) * 0.1))
                            x2 = min(image_w, int(x2 + (x2 - x1) * 0.1))
                            y1 = max(0, int(y1 - (y2 - y1) * 0.1))
                            y2 = min(image_h, int(y2 + (y2 - y1) * 0.1))

                            crop_image_queue.append(image[y1:y2, x1:x2])
                            crop_mask_queue.append(mask[y1:y2, x1:x2])
                            crop_bbox_queue.append([x1, y1, x2, y2])

                            crop_predict_sigmoid_queue.append(pred[y1:y2, x1:x2])  # [h,w,
                            crop_predict_logist_queue.append(torch.from_numpy(pred_raw).permute(1, 2, 0).numpy().astype(np.float32)[y1:y2, x1:x2])

                            pred_mask = (pred > 0.5).astype(np.float32)
                            crop_predict_queue.append(pred_mask[y1:y2, x1:x2])

            if args.calc_raw_dice:
                raw_dice=calc_dice(binary(raw_merge),mask)
                raw_dices.append(raw_dice)

            if args.dice_percent > 0:
                new_dice=calc_dice(binary(raw_merge),mask)
                flag=flag and (new_dice<=args.dice_percent)
                count=count+1 if flag else count

            for i in range(len(crop_image_queue)):
                crop_image = crop_image_queue[i]
                if args.prompt=="bbox" and args.four_channel:
                    print("use four channel")
                    fourth_channel = crop_predict_queue[i].astype(np.uint8) * 255
                    crop_image = np.concatenate([crop_image, fourth_channel], axis=2)
                height = crop_image.shape[0]
                width = crop_image.shape[1]

                optimizer.zero_grad()
                trans = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((args.crop_image_size, args.crop_image_size)),
                                            transforms.ToTensor()
                                            ])

                crop_image = trans(crop_image).unsqueeze(0).cuda()

                crop_mask = torch.from_numpy(crop_mask_queue[i]).permute(2, 0, 1)  # [1,h,w]
                crop_mask = crop_mask.unsqueeze(0)
                crop_mask = F.interpolate(crop_mask, mode="bilinear", size=(args.crop_image_size, args.crop_image_size))
                crop_mask = (crop_mask >= 0.5).float().cuda()

                if flag:
                    if len(branch_image) < args.nums:
                        branch_image.append(crop_image)
                        branch_mask.append(crop_mask)
                    else:
                        branch_image.pop(0)
                        branch_image.append(crop_image)
                        branch_mask.pop(0)
                        branch_mask.append(crop_mask)

                    crop_image_new = torch.cat(branch_image, dim=0)
                    output = net(crop_image_new)
                    temp_output = output[-1, :, :, :].unsqueeze(0)
                    temp_output = F.interpolate(temp_output, mode="bilinear", size=(height, width))
                    unet_predict_queue.append(temp_output.detach().cpu().sigmoid().squeeze(0).permute(1,2,0).numpy()) # 0~1  ndarray  [h,w,1]
                    unet_logist_queue.append(temp_output.detach().cpu().squeeze(0).permute(1,2,0).numpy())
                    crop_mask_new = torch.cat(branch_mask, dim=0)

                    loss = structure_loss(output, crop_mask_new)
                    loss.backward()
                    optimizer.step()
                else:
                    if len(branch_image) and len(branch_mask):
                        crop_image_new = torch.cat(branch_image, dim=0)
                        output = net(crop_image_new)
                        crop_mask_new = torch.cat(branch_mask, dim=0)
                        loss = structure_loss(output, crop_mask_new)
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        output = net(crop_image)
                        temp_output = output[-1, :, :, :].unsqueeze(0)
                        temp_output = F.interpolate(temp_output, mode="bilinear", size=(height, width))
                        unet_predict_queue.append(temp_output.detach().cpu().sigmoid().squeeze(0).permute(1, 2, 0).numpy())  # 0~1  ndarray  [h,w,1]
                        unet_logist_queue.append(temp_output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy())


            for i in range(len(crop_image_queue)):

                best_alpha=1
                best_dice=0
                for a  in [round(g,2) for g in  np.arange(0,args.alpha_percent+0.01,0.05).tolist()]:
                    temp = a *crop_predict_logist_queue[i]  + (1-a) * unet_logist_queue[i]
                    temp=torch.from_numpy(temp).sigmoid().numpy()
                    temp_dice=calc_dice(temp,crop_mask_queue[i])
                    # logger.info(f"{a}_{temp_dice}")

                    if temp_dice>best_dice:
                        best_alpha=a
                        best_dice=temp_dice
                # logger.info("_"*50)
                alphas.append(best_alpha)
                alpha_num=args.alpha_num
                if len(alphas)<=alpha_num:
                    temp_merge=crop_predict_sigmoid_queue[i]
                    mean_alphas.append(1)
                else:
                    mean_a=sum(alphas[-1-alpha_num:-1])/alpha_num
                    temp_merge=(mean_a) *crop_predict_logist_queue[i]  + (1-mean_a) * unet_logist_queue[i]
                    temp_merge = torch.from_numpy(temp_merge).sigmoid().numpy()
                    mean_alphas.append(mean_a)



                crop_merge_queue.append(binary(temp_merge))

                fix, ax = plt.subplots(2, 4)
                ax[0, 0].set_title("image")
                ax[0, 0].imshow(crop_image_queue[i])  # 原图
                ax[0, 0].axis("off")
                ax[0, 1].set_title("sam")
                ax[0, 1].imshow(crop_predict_sigmoid_queue[i], cmap="gray")  # sam图像
                ax[0, 1].axis("off")
                ax[0, 2].set_title("other_net")
                ax[0, 2].imshow(unet_predict_queue[i], cmap="gray")  # unet图像
                ax[0, 2].axis("off")
                ax[0, 3].set_title("final_result")
                ax[0, 3].imshow(temp_merge, cmap="gray")
                ax[0, 3].axis("off")
                ax[1, 0].set_title("mask")
                ax[1, 0].imshow(crop_mask_queue[i], cmap="gray")
                ax[1, 0].axis("off")
                ax[1, 1].set_title("sam_bin")
                ax[1, 1].imshow(binary(crop_predict_sigmoid_queue[i]), cmap="gray")
                ax[1, 1].axis("off")
                ax[1, 2].set_title("other_net_bin")
                ax[1, 2].imshow(binary(unet_predict_queue[i]), cmap="gray")
                ax[1, 2].axis("off")
                ax[1, 3].set_title("final_result_bin")
                ax[1, 3].imshow(binary(temp_merge), cmap="gray")
                ax[1, 3].axis("off")
                temp_sample_name = sample_name.split(".")[0] + ".jpg"
                plt.savefig(f"{view_path}/{i}_{temp_sample_name}")
                sub_sample_names.append(f"{i}_{temp_sample_name}")
                sub_parts.append(part)
                plt.close("all")


            for i in range(len(crop_image_queue)):
                x1, y1, x2, y2 = crop_bbox_queue[i]
                pre_merge[y1:y2, x1:x2, :] = binary(pre_merge[y1:y2, x1:x2, :] + crop_merge_queue[i])
            final_dice=calc_dice(pre_merge[:,:,0],mask[:,:,0])
            last["part"].append(part)
            last["sample_names"].append(sample_name)
            last["final_dice"].append(final_dice)
            cv2.imwrite(os.path.join(predict_path, sample_name), pre_merge*255)

        if args.dice_percent>0:
            gt_percent_queue.append(count/len(sample_list))
            logger.info(f"gt_percent_{part},{count/len(sample_list)}")

    if args.dice_percent>0:
        logger.info(f"total_gt_percent:,{sum(gt_percent_queue)/len(gt_percent_queue)}")
        logger.info(gt_percent_queue)
    alpha_dataframe=pd.DataFrame({"sub_parts":sub_parts,"sub_sample_names":sub_sample_names,"best_alphas":alphas,"mean_alphas":mean_alphas})
    save_alpha_path=f"{args.model_name}_{directory}_alpha.csv" if args.directory=="Polyp" else f"{args.model_name}_{directory}_{part}_alpha.csv"
    alpha_dataframe.to_csv(os.path.join(args.result_path, save_alpha_path),index=False)
    if args.calc_raw_dice:
        last["raw_dice"]=raw_dices
    final_dice_dataframe=pd.DataFrame(last)
    save_dice_path = f"{args.model_name}_{directory}_dice.csv" if args.directory == "Polyp" else f"{args.model_name}_{directory}_{part}_dice.csv"
    final_dice_dataframe.to_csv(os.path.join(args.result_path, save_dice_path),index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha_num",type=int,default=5,help="alpha nums")
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--nclasses", default=1, type=int)
    parser.add_argument("--calc_raw_dice",type=bool,default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--crop_image_size", type=int, default=128)
    parser.add_argument("--interval",type=int,default=1)
    parser.add_argument("--dice_percent",type=float,default=0)
    parser.add_argument("--nums", type=int, default=32, help="queue length")
    parser.add_argument("--alpha_percent", type=float, default=1, help="0~1")


    parser.add_argument("--model_name", default="MedSAM_bbox", type=str,help="choose from  SAM,MedSAM_bbox,MedSAM_point")
    parser.add_argument("--directory", type=str, default="Polyp")
    parser.add_argument("--parts", type=str,nargs="+",default=[ "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300" ])
    parser.add_argument("--prompt",type=str,default="bbox")
    parser.add_argument("--four_channel",action="store_true")

    # directory = "BUSI"
    # "benign","malignant"
    # directory = "GlaS"
    # "benign", "malignant"
    # directory="Polyp"
    # "CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
    # directory="fluidchallenge"
    # "cirrus","topcon","spectralis"

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    exp_path=f"exp_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    args.result_path = f"{args.result_path}/{exp_path}_{args.directory}_{args.model_name}_{args.alpha_percent}_{args.nums}_{args.prompt}_{args.dice_percent}"
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_path=f"{args.result_path}/log.txt")
    print_args(args,logger)

    predict_with_prompt(args=args, logger=logger, directory=args.directory, parts=args.parts)
    dice_and_hf(args, logger, args.directory, args.parts)
    print(exp_path)



