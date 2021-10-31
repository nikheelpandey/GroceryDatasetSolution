import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
sometimes = lambda aug: iaa.Sometimes(0.5, aug)


def augment(image,bounding_boxes):

    seq = iaa.Sequential([
        # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        # iaa.Affine(
        #     translate_px={"x": 40, "y": 60},
        #     scale=(0.7, 0.9)
        # ),
        iaa.Fliplr(0.8), # horizontally flip 80% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # iaa.SomeOf((0, 5),
        #     [
        #     iaa.AverageBlur(k=(2, 11)),

        #     iaa.Invert(0.05, per_channel=True), # invert color channels

        #     # Add a value of -10 to 10 to each pixel.
        #     iaa.Add((-10, 10), per_channel=0.5),

        #     # Change brightness of images (50-150% of original value).
        #     iaa.Multiply((0.5, 1.5), per_channel=0.5),

        #     # Improve or worsen the contrast of images.
        #     iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

        #     # Convert each image to grayscale and then overlay the
        #     # result with the original with random alpha. I.e. remove
        #     # colors with varying strengths.
        #     iaa.Grayscale(alpha=(0.0, 1.0)),
        #     ]),
        # # In some images move pixels locally around (with random
        # # strengths).
        


        # crop some of the images by 0-10% of their height/width
        
        ])

    return image, bounding_boxes




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
