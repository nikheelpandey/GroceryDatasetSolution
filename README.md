# GroceryDatasetSolution
This repository contains an implementation of SSD300.
The goal is to achieve maximum mAP while using a single aspect ratio per feature map. 

To download the dataset and set the repo structure: `sh data_preperation.sh`
 


Dataset Preparation: The file `data.py` contains the following class over which a dataloader can be fitted:



```
class ShelfImageDataset(Dataset):
    def __init__(self, df, image_path, train=True,return_orig=False):
        self.df = df
        self.train = train
        self.return_orig = return_orig
        self.image_path = image_path+'train/' if train else image_path+'test/'
        self.total_imgs = listdir(self.image_path)
        self.bbox_cols = ["x", "y","w","h"]
    def __len__(self):
        return len(self.total_imgs)

    # def _fix_df(self):
    #     list_images = listdir(self.image_path)
    #     self.df = self.df[self.df.names.isin(list_images)]
    #     self.df.reset_index(drop=True, inplace=True)
        
    def __getitem__(self, idx):

        
        img = self.total_imgs[idx]
        # print(idx, img)
        # print(self.image_path)
        orig_image = Image.open(self.image_path + img).convert('RGB')
        # print(type(orig_image))
        split_df = self.df[self.df["names"]==img]
        cls_ls, bbox = split_df["class"].values, split_df[self.bbox_cols].values
        


        aug_image = np.array(orig_image)
        aug_image = aug_image[:, :, ::-1].copy()

        aug_image,bbox = augment(np.array(aug_image), bbox)
        # (print(aug_image.shape))
        image = Image.fromarray(np.uint8(aug_image)).convert('RGB')
        # print(bbox)
        boxes = torch.tensor(bbox)
        # print(boxes.shape)
        boxes = utils.xywh_to_xyXY(boxes)
        # rescale image and boxes
        image, boxes = resize(orig_image, boxes, (300,300))
        # horizontal flip image and boxes with 80% prob (if it's a train sample)
        # if (uniform(0,1)>0.9 and self.train):
        #     image, boxes = hflip(image, boxes)
        image = imageTransforms(image)
        # image, boxes = korniaAug(image, boxes)
        label = torch.LongTensor([1]*boxes.size(0))
        if self.return_orig:
            return image, boxes, label, orig_image
        else:
            return image, boxes, label

```