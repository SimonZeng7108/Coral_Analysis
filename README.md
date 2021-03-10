# Coral Analysis via various methods
This repo showcases a various of tradiontial computer vision methods, unsupervised learning methods and deep learning models for coral skeletal image segmentation.<br/>
The dataset is downloaded from the original repo by [Ainsley Rutterford](https://github.com/ainsleyrutterford/deep-learning-coral-analysis/blob/master/README.md). 
## Traditional Computer Vision methods:
[Traditional CV/histogram_based.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/histogram_based.py): Multi-level histogram thresholding based segmentation<br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/histogram.png" width="300" height="300"><br/>

[Traditional CV/equalised_otsu.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/equalised_otsu.py): Binary Otsu's method <br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/otsu.png" width="300" height="300"><br/>

[Traditional CV/canny.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/canny.py): Canny's edge detection <br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Traditional%20CV/canny.png" width="300" height="300"><br/>

## Unsupervised learning methods:
[Unsupervised Learning/random_walker.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/random_walker.py): Random Walker Segmentaion <br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/random_walker.png" width="300" height="300"><br/>

[Unsupervised Learning/K-means-clustering.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/k_means_clustering.py): K-means clustering <br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/k_means_clustering.png" width="300" height="300"><br/>

[Unsupervised Learning/Gaussian_Mixture_Model.py](https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/Gaussian_Mixture_Model.py): Gaussian Mixture Model <br/>
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/image_denoised.jpg" width="300" height="300">
<img src="https://github.com/SimonZeng7108/Coral_Analysis/blob/main/Unsupervised%20Learning/GMM.png" width="300" height="300"><br/>

## Deep Learning Models:
<sup>This implementation is mostly based on my [Fetal_Segmentation](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch) repo.</sup>
<br/>


note: This UNet implementation is rather a vanilla model, there is no BatchNorm, DropOut utilised. If one follow the original paper strictly, there will be a conflict betweent input and output sizes(572 to 388). To avoid label and prediction mismatch in this implementatino, a resize function has been applied after every up-convolution in expansive path and at final output layer.<br/>

## Repository overview
[data/training/](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/tree/main/data/training): Stores all the unsplited training data <br/>
[data/test_set](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/tree/main/data/test_set): Stores all data for prediction use <br/>
[model/weights.pt](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/models/weights.pt): Stores the best weight model generated after training<br/>
[main.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/main.py): The main script imports dataset, trainer, loss functions to run the model <br/>
[dataset.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/dataset.py): Customise a dataset to process the trainig images <br/>
[model.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/model.py): Construct the SegNet and UNet model <br/>
[train.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/train.py): The trainer to run epochs <br/>
[loss_functions.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/loss_functions.py): Define the dice loss + BCElogits loss function <br/>
[predict.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/predict.py): Script to predict unlabeld images <br/>

## Requirements 
- `torch == 1.8.0`
- `torchvision`
- `torchsummary`
- `numpy`
- `scipy`
- `skimage`
- `matplotlib`
- `PIL`

## [main.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/main.py) workflow
The main.py file is the only file needs to be run and other utils will be import to here

```
Set Parameters
```
Set the paramers for path to `train/image`, `train/label`, `test/image`, `test/label` and `save_model`; also change the `h,w` for input image, which `model` to use, numbers of `epochs` to run, `batch_size`, `learning rate` and learning rate scheduler `dropping rate` for the optimizer

```
Data Augmentation
```
Defined a series of data transformation can be called upon dataloading

```
Dataset Loader
```
Call the customised `dataset.py` to meet pytorch `DataLoader` standard

```
Load Model
```
Load the prebuilt models as choice from parameters

```
Load Loss Function
```
Import the predefined `loss_function`, the loss function is the sum of BCELoss and Dice loss, the metrics is the Dice Coefficient.

```
Define Optimizer and Scheduler
```
An Adam optimiser is used with a learning rate scheduler when the loss plateaus


```
Trainer
```
load the pre-defined trainer function with parameteres set previously

```
Plots
```
Plot the graph for `Loss vs Epochs` and `Accuracy vs Epochs`

