
import os
import numpy as np
import torch

#Set parameters
path2predict="./data/predict_set/"
path2weights="./models/weights.pt"
h,w=256,256
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

imgsList=[pp for pp in os.listdir(path2predict)]
print("number of images:", len(imgsList))
rndImg=np.random.choice(imgsList , 1)
print(rndImg)

#Load model
from model import UNet
params_model={
        "input_shape": (1,h,w),
        "initial_filters": 16, 
        "num_outputs": 1,
            }
model = UNet()
model = model.to(device)

#Evaluate
from PIL import Image
from torchvision.transforms.functional import to_tensor
model.load_state_dict(torch.load(path2weights))
model.eval()

path2img = os.path.join(path2predict, rndImg[0])
img = Image.open(path2img)
img=img.resize((w,h))
img_t=to_tensor(img).unsqueeze(0).to(device)


pred=model(img_t)
print(pred.max())
pred=torch.sigmoid(pred)[0].detach().to('cpu')
# mask_pred= (pred[0]>=0.1).to('cpu')

#Plot the graph
#Define a show mask on image function
from torchvision.transforms.functional import to_pil_image
from skimage.segmentation import mark_boundaries
def show_img_mask(img, mask): 
    if torch.is_tensor(img):
        img=to_pil_image(img)
        mask=to_pil_image(mask)

    img_mask=mark_boundaries(
        np.array(img), 
        np.array(mask),
        outline_color=(0,1,0),
        color=(0,1,0))
    plt.imshow(img_mask)


import matplotlib.pylab as plt
plt.figure()
plt.subplot(1, 3, 1) 
plt.imshow(img, cmap="gray")

pred = to_pil_image(pred)
plt.subplot(1, 3, 2) 
plt.imshow(pred, cmap="gray")

plt.subplot(1, 3, 3) 
show_img_mask(img, pred)
plt.show()