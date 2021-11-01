import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
sometimes = lambda aug: iaa.Sometimes(0.5, aug)


def augment(image,bounding_boxes):
    # print()
    num_box,num_cord = bounding_boxes.shape[0],bounding_boxes.shape[1] 
    bounding_boxes = bounding_boxes.reshape(1,num_box,num_cord)

    seq = iaa.Sequential(
        [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.5), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        # percent=(-0.05, 0.1),
        # pad_mode=ia.ALL,
        # pad_cval=(0, 255)
        # )),
        # sometimes(iaa.Affine(
        # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        # rotate=(-45, 45), # rotate by -45 to +45 degrees
        # shear=(-16, 16), # shear by -16 to +16 degrees
        # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
        ],
        random_order=True
        )
    image, bounding_boxes = seq(image=image, bounding_boxes=bounding_boxes)
    return image, bounding_boxes.reshape(num_box,num_cord)




if __name__=="__main__":
    import pandas as pd
    import torch
    from data import ShelfImageDataset
    import cv2
    import numpy as np

    df = pd.read_csv("./annotations.csv",
            names=["names", "x", "y", "w", "h", "class"])


    dataset_tr = ShelfImageDataset(df, "./ShelfImages/", train=True,return_orig=True)
    image, boxes, _ , og_img = dataset_tr.__getitem__(2)
    # print(image.shape)
    img , bbox_mod = augment(image, boxes)

    open_cv_image = np.array(og_img)

    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    color = (0, 255, 0)  
    thickness = 2

    H,W,_ = open_cv_image.shape
    open_cv_image = cv2.resize(open_cv_image,(300,300))

    boxes = boxes.tolist()
    # print(boxes)

    for i in range(len(boxes)):

        x= (boxes[i][0]*300)
        y= (boxes[i][1]*300)
        X= (boxes[i][2]*300)-x
        Y= (boxes[i][3]*300)-y
        open_cv_image = cv2.rectangle(open_cv_image, (int(x),int(y)), (int(X),int(Y)), color, thickness)

    # open_cv_image = cv2.resize(open_cv_image,(W,H))
    cv2.imshow("",open_cv_image)
    cv2.waitKey(0)
