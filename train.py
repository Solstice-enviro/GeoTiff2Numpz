import torch
import albumentations
from albumentations.pytorch import ToTensorV2
    # Albumentations is a Python library for image augmentation.
    # Image augmentation is used in deep learning and computer vision
        # tasks to increase the quality of trained models.
    # The purpose of image augmentation is to create new training samples
        # from the existing data.
from tqdm import tqdm
    #tqdm derives from the Arabic word taqaddum (تقدّم) which can mean “progress,”
        # and is an abbreviation for “I love you so much” in Spanish (te quiero demasiado).
    #Instantly make your loops show a smart progress meter - just wrap any iterable with
        # tqdm(iterable), and you’re done!
import torch.nn as nn
import torch.optim as optim
from Unet2 import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
    #utils is a python file we create

#Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "D:/Programming/EuroSat/data/Training_128x128/images"
#TRAIN_MASK_DIR = ""
VAL_IMG_DIR = "D:/Programming/EuroSat/data/Validation_128x128/images"
TEST_IMG_DIR = "D:/Programming/EuroSat/data/Testing_128x128/images"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) #progress bar

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero.grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item)





def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizonalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
         [
             A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
             A.Normalize(
                 mean=[0.0, 0.0, 0.0],
                 std=[1.0, 1.0, 1.0],
                 max_pixel_value=255.0,
             ),
             ToTensorV2(),
         ],
     )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE) #change out_channels to the number of classes
    loss_fn = nn.BCEWithLogitsLoss() #need to change this to cross entropy loss to account for multiple classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint =  {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state.dict(),
        }
        save_checkpoint(checkpoint)

        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        #print some examples to a folder (to see if it looks good)
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )




if __name__ == "main": #On windows, need this to make sure we dont run into issues with NUM_WORKERS
    main()



