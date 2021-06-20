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
      
      ![image](https://user-images.githubusercontent.com/40360231/122669282-3a880300-d1da-11eb-8e6b-909db1d0ec60.png)
      
     
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

1. Our WNet model consists of two bridged UNet.The contracting and expanding path of each UNet contains 5 blocks,each having convolution layer with kernels size 3×3
   voxels,followed by residual block , Rectified Linear Units(ReLU) activation function and a pooling layer with stride 2. 

2. The residual block helps to preserve the location information of pixels while downsampling. It learns from the residue of true output and the input.
3. At the end sigmoid activation function is applied on the output layer. While training the model we included a dropout of 0.2 and Adam optimizer with learning rate of 1e-5.

## 6. Performance Metrics

Dice Coefficient and Dice Loss
	  	Dice Coefficient = (2 × | GT ∩ SEG | ) / (GT2 + SEG2 + ε)
Where, GT is the standard ground truth for brain tumor, SEG is the predicted segmented tumorous region and ε=1e-6.
  		LDice = 1 - Dice Coefficient

## 7. Experimentation and Results

1. Data split into 3:1:1- 60% images for training, 20% images for testing, and 20% for validation.
2. All the models are trained with batch size-8 and no of epochs-30

## 8. Conclusion

1. Several Deep Learning and Image processing techniques were studied in the due course of the project. 
2. We completed research work with current usage of techniques in image preprocessing, image segmentation, common feature extraction and classification recently used were
   analyzed and studied. We choose in total 3 systems-
      SOBEL + MD-UNET (Dice score-0.9918)
      2D -VNET (Dice score-0.9947)
      WNET (Dice score-0.9964)
3. The output we received is of tumor region highlighted into three regions which are edema, enhancing tumor and non enhancing tumor. We achieved the highest dice score of
    99.64% with WNET architecture on training the dataset through our system.
   




























