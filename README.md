# Brain-Tumor-Segmentation-Using-Deep-Neural-Networks

## Description
This project presents the use of deep learning and image processing techniques for the segmentation of tumors into various classes. We used the following three approaches for tumor segmentation.
1) The first approach is a hybrid system within which we use Sobel operator which is an Image Processing technique for edge clarification and a modified U-Net which is a Deep Learning Neural Network for training. 
2) The second approach proposes a V-Net model architecture for tumor segmentation.
3) The third approach proposes a W-Net model architecture for tumor segmentation.

BRATS 2018 dataset is used to train the model which consists of 3D MRIs of the brain. The output is a segmented image of a tumor consisting of three regions which are edema, enhancing tumor and non enhancing tumor.

## Table of Contents
1. Dataset.
2. Preprocessing.
3. First Approach: Sobel and modified U-Net. 
4. Second approach: V-Net.
5. Third Approach: W-Net.
6. Performance Metrics.
7. Experimentation and Results.
8. Conclusion.
9. GUI of the project.
10. How to run?

## 1. Dataset
1. Data released with multimodal brain tumor segmentation challenge by Medical Image Computing and Computer Assisted Intervention (MICCAI)
2. 3D dataset consists of pre surgical MRI scans of 210 High-Grade Glioma (HGG) patients and 75 Low-Grade Glioma (LGG).
3. 4 modalities- T1-weighted (T1w), post-contrast T1-weighted (T1ce), T2-weighted (T2), Fluid Attenuated Inversion Recovery (FLAIR).
4. Ground truth included
5. These images are manually segmented by expert neuroradiologist labelled as as enhancing tumor (label 4), peritumoral edema (label 2), and the core (label 1)
6. Presence of Multiple tumor region is more visible with HGG than LGG- thus only HGGs are used.

![image](https://user-images.githubusercontent.com/40360231/122669055-1972e280-d1d9-11eb-94b7-862e6f774ff4.png)


## 2. Pre-Processing

1. Each 3D volume -240×240×155
2. 3D image sliced- 155 slices
3. Only 90 slices from each volume selected( covering max tumor area)
4. These slices cropped to  192×192 (Background noise)
5. 18,900 2D images of each MRI modality
6. Total images-75,600

## 3. First Approach- Sobel and Modified Unet

1.EDGE FEATURE EXTRACTION USING SOBEL OPERATOR
      Image convolution along horizontal and vertical axis 
      
![image](https://user-images.githubusercontent.com/40360231/122669397-bf731c80-d1da-11eb-851c-29c526fddced.png)
      
     
  Why sobel?
     -Good smoothing effect to random noise
     -Better appearance of tumor region edges due to differential of two rows and two columns
2. MODEL DESCRIPTION
Encoding part- performs downsampling and Decoding part-upsampling.
Down sampling is for context(tumor in this case) and up sampling is for localization(position).
Modified deep U-shaped Network (md-UNET) with multiple skipped connections.

![image](https://user-images.githubusercontent.com/40360231/122669332-74590980-d1da-11eb-8794-27c2243056a2.png)


ENCODER PART

-Five blocks of consecutive convolutional layers.
-Each block has two 3×3 convolutional layers with same padding and activation function as ReLU (Rectified Linear Unit), each followed by a batch normalization layer and a
 maximum pooling layer.
-A dropout layer with dropout rate of 20% is placed after two blocks to avoid overfitting.
-6th block acts as a bridge between the encoding and the decoding part, which consists of a 3×3 convolutional layer with same padding and ReLU activation function

DECODER PART

-Decoder part too has 5 consecutive blocks.
-Each block has a convolutional transpose layer of 3×3 with same padding and ReLU activation function merged with skip connection from previous encoder part, followed by a
 convolution layer 3×3 with same padding and ReLU activation
-Dropout Layer present after two blocks.
-Output layer with softmax activation function used for classification.

SUMMARIZING THE MODEL

1. No of activation filters used are 32, 64, 128, 256, 512 in five consecutive blocks of both encoder and decoder.
2. 1024 activation kernels at the center.
3. 10 Blocks
4. 21 Convolutional layers
5. 10 layers of batch normalization 
6. 5 layers of maximum pooling and there is finally one convolutional layer at the bridge point of U-shaped network.
7. ptimizer used- ADAM.
8. Loss function- DICE.

## 4. Second approach VNET

![image](https://user-images.githubusercontent.com/40360231/122669445-ffd29a80-d1da-11eb-9a1c-1a6c7cb55e0a.png)


1. The VNet model consists of contracting and expanding paths like the UNet model. 
2. The downsampling and upsampling path consist of 5 residual blocks, each consisting of 1 to 3 convolutional layers.
3. Each convolutional layer at downsampling residual block uses kernels having size 5×5 voxels with stride 2, followed by batch normalization and activation function Parametric 
   Rectified Linear Unit(PReLU). 
4. Addition operation is performed at the end of each residual block to add the features.
5. As fusion of features with addition operation changes the distribution of weights,this method performs better for our model as compared to concatenation operation. 
6. Downsampling reduces the input size to half and doubles the number of features at each block.
7. At the upsampling side,each block contains a deconvolution layer which maps to the features extracted by the respective downsampling block producing outputs of the same size
   as the input volume.
8. Finally softmax activation function is applied on the output layer.
9. We have used Adam optimizer with a learning rate of 2e-4 while training this model.

## 5.Third Approach WNET

![image](https://user-images.githubusercontent.com/40360231/122669458-18db4b80-d1db-11eb-9a1b-e58b65520d88.png)


1. Our WNet model consists of two bridged UNet.The contracting and expanding path of each UNet contains 5 blocks,each having convolution layer with kernels size 3×3
   voxels,followed by residual block , Rectified Linear Units(ReLU) activation function and a pooling layer with stride 2. 

2. The residual block helps to preserve the location information of pixels while downsampling. It learns from the residue of true output and the input.

![image](https://user-images.githubusercontent.com/40360231/122669488-41fbdc00-d1db-11eb-943d-5b00ffb4b0dc.png)


3. At the end sigmoid activation function is applied on the output layer. While training the model we included a dropout of 0.2 and Adam optimizer with learning rate of 1e-5.

## 6. Performance Metrics

Dice Coefficient and Dice Loss
	  	Dice Coefficient = (2 × | GT ∩ SEG | ) / (GT2 + SEG2 + ε)
Where, GT is the standard ground truth for brain tumor, SEG is the predicted segmented tumorous region and ε=1e-6.
  		LDice = 1 - Dice Coefficient

## 7. Experimentation and Results

1. Data split into 3:1:1- 60% images for training, 20% images for testing, and 20% for validation.
2. All the models are trained with batch size-8 and no of epochs-30

![image](https://user-images.githubusercontent.com/40360231/122669512-635cc800-d1db-11eb-8996-9f80a6e3945c.png)

![image](https://user-images.githubusercontent.com/40360231/122670113-e717b400-d1dd-11eb-9ac1-883e4e92fd52.png)



## 8. Conclusion

1. Several Deep Learning and Image processing techniques were studied in the due course of the project. 
2. We completed research work with current usage of techniques in image preprocessing, image segmentation, common feature extraction and classification recently used were
   analyzed and studied. We choose in total 3 systems-
      SOBEL + MD-UNET (Dice score-0.9918)
      2D -VNET (Dice score-0.9947)
      WNET (Dice score-0.9964)
3. The output we received is of tumor region highlighted into three regions which are edema, enhancing tumor and non enhancing tumor. We achieved the highest dice score of
    99.64% with WNET architecture on training the dataset through our system.
    
 
## 9. GUI of Project

![image](https://user-images.githubusercontent.com/40360231/122669670-12010880-d1dc-11eb-8845-bd2b2aef5b00.png)

Link to website: https://bts-seg.anvil.app/

## 10. How to Run?

![image](https://user-images.githubusercontent.com/40360231/122670180-407fe300-d1de-11eb-9769-2856ae3d07ab.png)


### Description and objective of the GUI 

The GUI consists of an anvil-app that acts as a client and a Google colaboratory notebook that acts as a server (back-end) for the anvil app. The anvil app takes the 3D MRI nifti (.nii.gz) files of 4 modalities (Flair, T1, T1ce, T2) and slice no. and passes them to server. The server downloads the deep learning models, do predictions of that particular slice and sends the image of slices of 4 modalities as well as the prediction of tumor region using 4 different models (U-Net, sobel and modified U-Net, V-Net and W-Net) to frontend. The anvil app (front-end) displays the images. 

#### Following steps are performed to perform the above task 
1) Client: Takes four modalities 3D MRI files and slice no. as input and pass them to server 
2) Server: 
a) The server downloads the pre-trained deep learning models using wget. b) The server is linked to an anvil app. 
c) The server receives the four modality files and slice no. 
d) It extracts the 2D slices of slice no from each of the 4 modalities and also crop them to a shape of 192 X 192. 
e) Apply sobel operator on the slices. 
f) The slices on which sobel operator is applied are given to Sobel-Modified U-Net model whereas the original 2D slices are remaining models (U-Net, V-Net, W-Net) for prediction. 
g) Save the generated 4 predicted images using different models (U-Net, sobel and modified U-Net, V-Net and W-Net) and also slices of given slice no. of 4 modalities(Flair, T1, T1ce, T2). 
h) Pass all the 8 images to client anvil-app. 
3) Client: Display all the received images at the front-end. 

#### How to Run? 
1) Download any HGG patient's data (4 modalities' MRI scan) from BraTS 2018 dataset or from the link. 
2) Run the Google colaboratory as it acts as a server for this anvil project. If any error occurs do factory reset runtime and run again. Ensure that all cells are executed without any errors before going to next step. 
3) Go to link and upload the Flair, T1, T1-ce, T2 3D nifti (.nii.gz) files. Enter slice no. as the models are trained on 2D images and predictions are carried out for that particular slice only. The slice no. is expected to be in between 30 and 119 (both 2 included). Submit the data and wait for sometime as prediction is going on in the backend. 
4) You can view Flair, T1, T1ce, T2 slices of that particular slice no. entered and also the prediction of tumor regions using U-Net, Sobel Operator and modified U-Net model, V-Net, W-Net models. 

### Description and objective of prediction using MD-Net With Sobel Operator 
This module takes HGG dataset of first 50 patients as input and generates a model for prediction of tumor region. It shows predictions of some sample images and displays the dice score and accuracy of the model. The model is then saved in google drive. 

#### Following steps are performed to perform the above task 
1) Load the HGG images of BRaTS 2018 dataset. 
2) Slice the images, take central 90 slices (30-120) and eliminate rest of them. 3) Crop the images to eliminate the background. 
4) Save the generated numpy images so that it can be used for further models. 5) Pass all the images through Sobel Operator for edge clarification. 6) Create Modified deep U-shaped Net(MD-Unet) model. 
7) Train the U-Net model using cropped images, with dice loss as a loss function and dice coefficient, no of epochs = 30, batch size = 8. Optimizer used is Adam optimizer with learning rate of 10^-5. 
8) Show some predictions, accuracy and loss graph. 
9) Save the trained U-Net model in Google drive. 

#### How to Run? 
1) Go to dataset link and add this folder as a shortcut to drive. 
2) Run the collaboratory project. Also, mount the google drive when asked. 
3) 3) The U-Net Model will be saved in your google drive. 

### Description and objective of prediction using V-Net Model 
This module takes HGG dataset of first 50 patients as input and generates a V-Net model for prediction of tumor region. It shows predictions of some sample images and displays the dice score and accuracy of the model. The model is then saved in google drive

#### Following steps are performed to perform the above task 
1) Load the HGG images of BRaTS 2018 dataset. 
2) Slice the images, take central 90 slices (30-120) and eliminate rest of them. 6) Crop the images to eliminate the background. 
3) Create the V-Net Model. 
4) Train the V-Net model using cropped images, with dice loss as a loss function and dice coefficient, no of epochs = 30, batch size = 8. Optimizer used is Adam optimizer with learning rate of 10^-5. 
5) Show some predictions, accuracy and loss graph. 
6)Save the trained V-Net model in Google drive

#### How to Run?
1) Go to dataset link and add this folder as a shortcut to drive. 
2) Run the colaboratory project. Also, mount the google drive when asked. 
3) 3) The U-Net Model will be saved in your google drive. 

### Description and objective of prediction using W-Net Model 
This module takes HGG dataset of first 50 patients as input and generates a V-Net model for prediction of tumor region. It shows predictions of some sample images and displays the dice score and accuracy of the model. The model is then saved in google drive. 

#### Following steps are performed to perform the above task 
1) Load the HGG images of BRaTS 2018 dataset. 
2) Slice the images, take central 90 slices (30-120) and eliminate rest of them. 3) Crop the images to eliminate the background. 
4) Create the W-Net Model. 
5) Train the W-Net model using cropped images, with dice loss as a loss function and dice coefficient, no of epochs = 30, batch size = 8. Optimizer used is Adam optimizer with learning rate of 10^-5. 
6) Show some predictions, accuracy and loss graph. 
7) Save the trained W-Net model in Google drive.

#### How to Run?
1) Go to dataset link and add this folder as a shortcut to drive. 
2) Run the colaboratory project. Also, mount the google drive when asked. 3) The U-Net Model will be saved in your google drive. 



   




























