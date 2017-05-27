# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/label_histos.png "Visualization"

[image2]: ./writeup_images/signs_gray.png "Grayscaling"
[image3]: ./writeup_images/signs_normalized.png "Normalized"
[image4]: ./writeup_images/signs_color.png "Color"

[image5]: ./writeup_images/german_traffic_signs.png "German traffic signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/fjp/CarND-Traffic-Sign-Classifier-P2)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 traffic signs
* The size of the validation set is 4410 traffic signs
* The size of test set is 12630 traffic signs
* The shape of a traffic sign image is (height = 32, width = 32, color channels = 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the labels of the training, validation and test sets are distributed. They seem equally distributed across the different sets but not all signs are equally likely.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here are the original traffic sign images

![alt text][image4]

As a first step, I decided to convert the images to grayscale because it lowers the computational cost required.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it makes the life easier for the optimizer because the bias is initialized with zero.

![alt text][image3]

I did not decided to generate additional data because the accuracy seemed fine. However, a possible augmentation of the data includes: blurring, mirroring, changing the brightness, ...


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the lenet architecture from the Udacity lecture and added a drop out layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized grayscale image  			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| ReLu Activation function						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| ReLu Activation function						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| outpu 400    									|
| Flatten       		| outpu 120    									|
| RELU					| ReLu Activation function						|
| Dropout               | Dropout layer with specific keep probability  |
| Flatten       		| outpu 84    									|
| RELU					| ReLu Activation function						|
| Dropout               | Dropout layer with specific keep probability  |
| Fully connected		| outpu 43    									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer from the udacity lectures. 
- BATCH_SIZE = 100
- EPOCHS = 60

**Hyperparameters**
- mu = 0
- sigma = 0.1
- low learning rate of 0.0009
- dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.952
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

    The LeNet Lab from the Udacity lectures.
    
    
* What were some problems with the initial architecture?

Decreasing accuracy but this was due to wrong normalization.
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I just added two drop out layers because this realatively new technique was suggested in the lecture.

* Which parameters were tuned? How were they adjusted and why?

I used a relatively small batch size because I trained the model on a local machine instead of the cloud

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
Similar architecture to LeNet, just added drop out layers.
* Why did you believe it would be relevant to the traffic sign application?
Because the LeNet architecture can be used to classify general images of size 32x32.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy was 0.952 and the test set 0.944 which is above the requirements. However, the accuracy could be increased by augmenting the data sets.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

Unfortunately the sizes were different. Therefore, I transformed the images using cv2.resize to obtain 32x32 images. Then I grayscaled and normalized the images. 

Due to the scaling the 30 km/h speed limit was hard to classify which led to to wrong classification. The other images were classified correctly.

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
# General caution, Speed limit (30), Yield, Stop, Road work

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution  		| General caution								| 
| Speed limit (30)  	| Road Work										|
| Yield					| Yield											|
| Stop	      		    | Stop			         		 				|
| Road Work		        | Road Work      			     				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The 30 km/h speed limit was the only sign that did not get classified correctly. The reason is probably the scaling of the original traffic sign and that I did not augment the data set with scaled, rotated, ... images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

There seems to be something wrong. The model is 100 % certain that the top guess is correct. This is also true for all signs except the speed limit (30). The certainty on the top guess leads to 0.0 probability of all other guesses.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution   					    	| 
| 1.0     				| Speed limit (30) 					    		|
| 1.0					| Yield											|
| 1.0	      			| Stop					            			|
| 1.0				    | Road Work      	     						|
