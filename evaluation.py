from tqdm import tqdm
import time
import utils
import torch
from ssd_config import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from torch.optim import SGD
from ssd import SSD, MultiBoxLoss
from ssd_utils import calc_mAP
import pandas as pd
from trainer import eval
import numpy as np

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2
from ssd_utils import calc_mAP
import os


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    # torch.cuda.set_device(device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")



fname = "checkpoint_ssd_imgaug_.300.1.1"

config = SSDConfig()
# device = "cpu"#config.DEVICE

torch.manual_seed(config.seed)



config.PATH_TO_ANNOTATION="./annotations.csv"
config.PATH_TO_IMAGES = "ShelfImages/"

config.PATH_TO_CHECKPOINT = "ckpt/"+fname+".pth.tar"



df = pd.read_csv(config.PATH_TO_ANNOTATION,
            names=["names", "x", "y", "w", "h", "class"])



checkpoint = torch.load(config.PATH_TO_CHECKPOINT , map_location=torch.device("cuda"))
model = checkpoint['model'].to(device)
model.config.DEVICE = torch.device("cuda")



# class UnNormalize(object):
#     def __init__(self):
        
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return tensor





model.eval()

"""
uncomment the following for calculating mAP on test set
"""

# min_score=0.25
# max_overlap=0.2


# dataloader
# dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False,return_orig=True)
# dataloader = DataLoader(dataset,
#                         shuffle=False,
#                         collate_fn=collate_fn,
#                         batch_size=2,
#                         num_workers=config.NUM_DATALOADER_WORKERS)


# dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False,return_orig=True)
# gt_boxes = []
# gt_labels = []
# pred_boxes = []
# pred_labels = []
# pred_scores = []
# for batch in dataloader:
#     orig_images = batch[0].to(device)
#     orig_boxes = [b.to(device) for b in batch[1]] 
#     orig_labels = [l.to(device) for l in batch[2]]
#     with torch.no_grad():
#         loc_gcxgcy, scores = model(orig_images)
#         boxes, labels, scores = model.detect_objects(loc_gcxgcy, scores, min_score=min_score, max_overlap=max_overlap)
#     gt_boxes.extend(orig_boxes)
#     gt_labels.extend(orig_labels)
#     pred_boxes.extend(boxes)
#     pred_labels.extend(labels)
#     pred_scores.extend(scores)
# AP, mAP, Recall = calc_mAP(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)


# print(f"Avg Precision: {AP.item()}")
# print(f"MeanAvg Precision: {mAP}")
# print(f"Avg Recall: {torch.mean(Recall).item()}")





min_score=0.25
max_overlap=0.5


dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False,return_orig=True)

for k in tqdm(range(len(os.listdir("ShelfImages/test/")))):
    img_tensor, og_bboxes, og_labels, og_image = dataset.__getitem__(k)
    
    loc_gcxgcy, scores = model(img_tensor.view(-1,3,300,300).to(device))
    boxes, labels, scores = model.detect_objects(loc_gcxgcy, scores, min_score=min_score, max_overlap=max_overlap)
    
    
    

    open_cv_image = np.array(og_image)

    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    color = (0, 255, 0)  
    thickness = 2

    H,W,_ = open_cv_image.shape
    open_cv_image = cv2.resize(open_cv_image,(300,300))

    boxes = boxes[0].tolist()
#     print(boxes)

    for i in range(len(boxes)):

        x= (boxes[i][0]*300)
        y= (boxes[i][1]*300)
        X= (boxes[i][2]*300)-x
        Y= (boxes[i][3]*300)-y
        open_cv_image = cv2.rectangle(open_cv_image, (int(x),int(y)), (int(X),int(Y)), color, thickness)

    open_cv_image = cv2.resize(open_cv_image,(W,H))
    cv2.imwrite("results/"+str(k)+".jpg", open_cv_image)

    del boxes, labels, scores , img_tensor, og_bboxes, og_labels, og_image, loc_gcxgcy
    
