import numpy as np
import pandas as pd
from PIL import Image
import os, sys, glob, shutil, json
from collections import Counter
from PIL import ImageEnhance
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imageio

#read files
train_path = glob.glob('./train_set/image/*.png')
label_path = glob.glob('./train_set/label/*.png')



count = 22
for i in range(332):
    if i <= 10:
        for j in range(166):
            print(i)
            img=Image.open(train_path[i])
            img = img.resize((256,256))
            labeled_image=Image.open(label_path[i])
            labeled_image = labeled_image.resize((256,256))    

            #genereates random numbers for transformations
            ramdon_1 = np.random.uniform(-1, 1, (256,256))
            ramdon_2 = np.random.uniform(0, 4)
            ramdon_3 = np.random.uniform(10, 350)

            #random rotations
            img = img.rotate(ramdon_3)
            label = labeled_image.rotate(ramdon_3)

            #random flips
            if  1 > ramdon_2 >= 0:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            elif  2 > ramdon_2 >= 1:
                img = img.transpose(Image.TRANSPOSE )
                label = label.transpose(Image.TRANSPOSE )
            elif  3 > ramdon_2 >= 2:
                img = img.transpose(Image.TRANSVERSE )
                label = label.transpose(Image.TRANSVERSE )
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

            #random contrast 
            ramdon_Contrast = np.random.uniform(0.8, 2)
            enh_con = ImageEnhance.Contrast(img)  
            img = enh_con.enhance(ramdon_Contrast)         
            #random brightness
            ramdon_Brightness = np.random.uniform(0.8, 1.5)
            enh_con = ImageEnhance.Brightness(img) 
            img = enh_con.enhance(ramdon_Brightness)              
            #random sharpeness
            ramdon_Sharpness = np.random.uniform(0.8, 2)
            enh_con = ImageEnhance.Sharpness(img)
            img = enh_con.enhance(ramdon_Sharpness)  
            #random noise
            img = img + ramdon_1
            img = Image.fromarray(np.uint8(img))  
    
            label = np.array(label)/255
            filter_h = 2
            filter_w = 2
            for h in range(0,len(label)-filter_h,filter_h):
                for w in range(0,len(label[0])-filter_w,filter_w):
                    conv = label[h:h+filter_h,w:w+filter_w]
                    if conv.sum() >= 1:
                        label[h:h+filter_h,w:w+filter_w] = 1
            label = Image.fromarray(np.uint8(label*255))  
            
            #save images
            img.save('./image_gen/'+ str(count) +".png")
            label.save("./label_gen/"+ str(count) +".png")  
            
            
            img = cv2.imread(filename='./image_gen/'+ str(count) +'.png')[:,:,:1]
            label = cv2.imread(filename='./label_gen/'+ str(count) +'.png')[:,:,:1]
            img = np.expand_dims(img, axis=0).astype(np.float32)
            label = np.expand_dims(label, axis=0).astype(np.int32)
            
            #Transformations
            tra = iaa.ElasticTransformation(alpha=2.0, sigma=1)
            img, label = tra(images=img, segmentation_maps=label)
            
            
            tra = iaa.PiecewiseAffine(scale=(0, 0.075), nb_rows=4, nb_cols=4, cval=0)
            img, label = tra(images=img, segmentation_maps=label)  
            
            
            tra = iaa.GaussianBlur(sigma=(0, 0.1))
            img, label = tra(images=img, segmentation_maps=label)              

            img = np.reshape(img,(256,256))
            label = np.reshape(label,(256,256))
            img = Image.fromarray(np.uint8(img))  
            label = Image.fromarray(np.uint8(label))  
            img.save('./image_gen/'+ str(count) +'.png')
            label.save('./label_gen/'+ str(count) +'.png') 
            
            count = count +1
    else:
        for j in range(5):
            print(i)
            img=Image.open(train_path[i])
            img = img.resize((256,256))
            labeled_image=Image.open(label_path[i])
            labeled_image = labeled_image.resize((256,256))    
            
            ramdon_1 = np.random.uniform(-1, 1, (256,256)) 
            ramdon_2 = np.random.uniform(0, 4) 
            ramdon_3 = np.random.uniform(10, 350) 
            
            img = img.rotate(ramdon_3)
            label = labeled_image.rotate(ramdon_3)
            
            if  1 > ramdon_2 >= 0:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            elif  2 > ramdon_2 >= 1:
                img = img.transpose(Image.TRANSPOSE )
                label = label.transpose(Image.TRANSPOSE )
            elif  3 > ramdon_2 >= 2:
                img = img.transpose(Image.TRANSVERSE )
                label = label.transpose(Image.TRANSVERSE )
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)            

            ramdon_Contrast = np.random.uniform(0.8, 2)
            enh_con = ImageEnhance.Contrast(img)  
            img = enh_con.enhance(ramdon_Contrast)         
            
            ramdon_Brightness = np.random.uniform(0.8, 1.5)
            enh_con = ImageEnhance.Brightness(img) 
            img = enh_con.enhance(ramdon_Brightness)              
            
            ramdon_Sharpness = np.random.uniform(0.8, 2)
            enh_con = ImageEnhance.Sharpness(img)
            img = enh_con.enhance(ramdon_Sharpness)  
            
            img = img + ramdon_1
            img = Image.fromarray(np.uint8(img))  
    
            label = np.array(label)/255
            filter_h = 2
            filter_w = 2
            for h in range(0,len(label)-filter_h,filter_h):
                for w in range(0,len(label[0])-filter_w,filter_w):
                    conv = label[h:h+filter_h,w:w+filter_w]
                    if conv.sum() >= 1:
                        label[h:h+filter_h,w:w+filter_w] = 1
            label = Image.fromarray(np.uint8(label*255))  
            
            img.save('./image_gen/'+ str(count) +".png")
            label.save("./label_gen/"+ str(count) +".png")  
            
            
            img = cv2.imread(filename='./image_gen/'+ str(count) +'.png')[:,:,:1]
            label = cv2.imread(filename='./label_gen/'+ str(count) +'.png')[:,:,:1]
            img = np.expand_dims(img, axis=0).astype(np.float32)
            label = np.expand_dims(label, axis=0).astype(np.int32)
            
            tra = iaa.ElasticTransformation(alpha=2.0, sigma=1)
            img, label = tra(images=img, segmentation_maps=label)
            
            tra = iaa.PiecewiseAffine(scale=(0, 0.075), nb_rows=4, nb_cols=4, cval=0)
            img, label = tra(images=img, segmentation_maps=label)  
            
            tra = iaa.GaussianBlur(sigma=(0, 0.1))
            img, label = tra(images=img, segmentation_maps=label)              
                        
            img = np.reshape(img,(256,256))
            label = np.reshape(label,(256,256))
            img = Image.fromarray(np.uint8(img))  
            label = Image.fromarray(np.uint8(label))  
            img.save('./image_gen/'+ str(count) +'.png')
            label.save('./label_gen/'+ str(count) +'.png') 
            
            count = count +1    


'''

img=Image.open('./test_result/result_0.png')
print(np.array(img)/255)
#enh_con = ImageEnhance.Contrast(img)  
#img_contrasted = enh_con.enhance(2.5) 

enh_con = ImageEnhance.Contrast(img)
img_contrasted = enh_con.enhance(100)  
img_contrasted.save("./2.5.png")

'''


