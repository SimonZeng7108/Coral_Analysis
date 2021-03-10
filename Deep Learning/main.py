
import os
import torch
from model import UNet, SegNet
#Set parameters
path2train_image="./data/train_set/image" #Path of train image
path2train_label="./data/train_set/label" #Path of train label
path2test_image="./data/test_set/image" #Path of test image
path2test_label="./data/test_set/label" #Path of test label

path2models= "./models/"            #Path to save best weight
h,w= 256,256                        #Input shape
model = UNet()                      #Options: UNet(), SegNet()
epochs = 150                        #Number of training iterations
batch_size = 8                      #Set batch size for dataloader
lr=1e-5                             #Optimiser learning rate
factor=0.5                          #Schedule learning rate drop rate


#Define data augmentation
from albumentations import (Rotate, HorizontalFlip, VerticalFlip, Compose, Resize)
transform_train = Compose([
    Resize(h,w), 
    Rotate(limit=35, p=0.5),
    HorizontalFlip(p=0.5), 
    VerticalFlip(p=0.5),
])
transform_test = Resize(h,w)

#Load dataset
from dataset import coral_dataset
train_ds = coral_dataset(path2train_image, path2train_label, transform=transform_train)
test_ds = coral_dataset(path2test_image, path2test_label, transform=transform_test)

#Create dataLoader
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=False) 

#Load model
import torch
from torchsummary import summary
model = model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
summary(model, input_size=(1, h, w))

#Load Loss Function
from loss_functions import loss_func

#Define Optimizer and Scheduler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
opt = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=factor, patience=10,verbose=1)

#Load trainer
from train import train_val
if not os.path.exists(path2models):
        os.mkdir(path2models)
params_train={
    "num_epochs": epochs,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": test_dl,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights.pt",
}
model, loss_hist, metric_hist=train_val(model,params_train)

#Plot graph
import matplotlib.pylab as plt
num_epochs=params_train["num_epochs"]
# plot loss progress
plt.figure(1)
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
# plot accuracy progress
plt.figure(2)
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

