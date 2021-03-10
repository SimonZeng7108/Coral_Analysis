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
This UNet implementation is rather a vanilla model, there is no BatchNorm, DropOut utilised. If one follow the original paper strictly, there will be a conflict betweent input and output sizes(572 to 388). To avoid label and prediction mismatch in this implementatino, a resize function has been applied after every up-convolution in expansive path and at final output layer.<br/>
<sup>This implementation is mostly based on my [Fetal_Segmentation](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch) repo.</sup>
<br/>





