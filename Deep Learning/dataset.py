#creating Custom Dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset


class coral_dataset(Dataset):
    def __init__(self, path2image, path2label, transform=None):      

        imgsList=[pp for pp in os.listdir(path2image)]
        anntsList=[pp for pp in os.listdir(path2label)]

        self.path2imgs = [os.path.join(path2image, fn) for fn in imgsList] 
        self.path2annts= [os.path.join(path2label, fn) for fn in anntsList]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)

        path2annt = self.path2annts[idx]
        mask = Image.open(path2annt)    
        
        image = np.array(image)
        mask = np.array(mask)
        mask=mask.astype("uint8")        

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']            

        image = to_tensor(image)            
        mask = to_tensor(mask)            
        return image, mask


# #show sample image
# path2train_image="./data/train_set/image" #Path of train image
# path2train_label="./data/train_set/label" #Path of train label
# path2test_image="./data/test_set/image" #Path of test image
# path2test_label="./data/test_set/label" #Path of test label

# dataset = coral_dataset(path2test_image, path2test_label)
# #Create DataLoader
# from torch.utils.data import DataLoader
# train_dl = DataLoader(dataset, batch_size=8, shuffle=False)

# for img_b, mask_b in train_dl:
#     image, mask = img_b[1], mask_b[1]
#     break

# import torch
# from torchvision.transforms.functional import to_pil_image
# from skimage.segmentation import mark_boundaries
# def show_img_mask(img, mask): 
#     if torch.is_tensor(img):
#         img=to_pil_image(img)
#         mask=to_pil_image(mask)

#     img_mask=mark_boundaries(
#         np.array(img), 
#         np.array(mask),
#         outline_color=(0,1,0),
#         color=(0,1,0))
#     plt.imshow(img_mask)

# image=to_pil_image(image)
# mask=to_pil_image(mask)


# import matplotlib.pylab as plt
# plt.figure('demo image')
# plt.subplot(1, 3, 1) 
# plt.imshow(image, cmap="gray")

# plt.subplot(1, 3, 2) 
# plt.imshow(mask, cmap="gray")

# plt.subplot(1, 3, 3) 
# show_img_mask(image, mask)
# plt.show()




