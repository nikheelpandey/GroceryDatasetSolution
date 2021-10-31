import utils
import torch
from ssd_config import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from torch.optim import SGD
from ssd import SSD, MultiBoxLoss
from trainer import train, eval
import pandas as pd
from tqdm import tqdm 
from datetime import datetime
import os

# from tensorboardX import SummaryWriter
from logger import Logger
fname = "checkpoint_ssd_.300.2.4"
log_dir = "runs/"+fname#str(datetime.now().strftime('%m%d%H%M%S'))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)


data_dict = {}
data_dict["training_loss"]= 100.0 
data_dict["mAP"] = 0.0
data_dict["Recall"] = 0.0


config = SSDConfig()
device = config.DEVICE

torch.manual_seed(config.seed)



config.PATH_TO_ANNOTATION="./annotations.csv"
config.PATH_TO_IMAGES = "ShelfImages/"

config.PATH_TO_CHECKPOINT = "ckpt/"+fname+".pth.tar"

config.PRINT_FREQ = 1
config.VGG_BN_FLAG = False
config.TRAIN_BATCH_SIZE = 4
config.LEARNING_RATE = 0.01
config.USE_PRETRAINED_VGG = True
config.NUM_ITERATIONS_TRAIN = 8000 
set_ratio = 1.5
config.FM_ASPECT_RATIO = [[set_ratio],
                        [set_ratio],
                        [set_ratio],
                        [set_ratio],
                        [set_ratio],
                        [set_ratio]]


df = pd.read_csv(config.PATH_TO_ANNOTATION,
            names=["names", "x", "y", "w", "h", "class"])



# dataloader
dataset_tr = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=True)
dataloader_tr = DataLoader(dataset_tr,
                           shuffle=True,
                           collate_fn=collate_fn,
                           batch_size=config.TRAIN_BATCH_SIZE,
                           num_workers=config.NUM_DATALOADER_WORKERS,
                           )

dataset_te = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False)
dataloader_te = DataLoader(dataset_te,
                           shuffle=True,
                           collate_fn=collate_fn,
                           batch_size=config.TRAIN_BATCH_SIZE,
                           num_workers=config.NUM_DATALOADER_WORKERS,
                           )

try:
    checkpoint = torch.load(config.PATH_TO_CHECKPOINT)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
except FileNotFoundError:
    print('PATH_TO_CHECKPOINT not specified in SSDConfig.\nMaking new model and optimizer.')
    start_epoch = 0
    model = SSD(config)
    model_parameters = utils.get_model_params(model)
    optimizer = SGD(params=[{'params': model_parameters['biases'], 'lr': 2 * config.LEARNING_RATE},
                        {'params': model_parameters['not_biases']}],
                        lr=config.LEARNING_RATE,
                        momentum=config.MOMENTUM,
                        weight_decay=config.WEIGHT_DECAY)

model.to(device)
criterion = MultiBoxLoss(model.priors_cxcy, config).to(device)
epochs = config.NUM_ITERATIONS_TRAIN // len(dataloader_tr)
decay_at_epoch = [int(epochs*x) for x in config.DECAY_LR_AT]


for epoch in tqdm(range(start_epoch, epochs)):
    if epoch in decay_at_epoch:
        utils.adjust_learning_rate(optimizer, config.DECAY_FRAC)
    
    loss = train(dataloader_tr, model, criterion, optimizer, epoch)
    
    data_dict["training_loss"] = loss
    
    if (epoch%5 == 0):
        print('Model checkpoint.', end=' ' )
        utils.save_checkpoint(epoch, model, optimizer, config, config.PATH_TO_CHECKPOINT)
        print('Model Evaluation.', end=' :: ')
        mAP, Recall = eval(model, dataloader_te, 0.6, 0.4)
        data_dict['mAP'] = mAP
        data_dict['Recall'] = (torch.mean(Recall).item()).item()
        
        print('mAP: ', mAP )
        print("Recall: ",Recall)


        utils.save_checkpoint(epoch, model, optimizer, config, config.PATH_TO_CHECKPOINT)

    logger.update_scalers(data_dict)
    