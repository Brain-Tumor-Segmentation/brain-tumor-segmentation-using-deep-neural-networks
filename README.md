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
11. Footnotes.
12. Contibutors

## 1. Dataset
1. Data released with multimodal brain tumor segmentation challenge by Medical Image Computing and Computer Assisted Intervention (MICCAI) is used for training.
2. 3D dataset consists of pre surgical MRI scans of 210 High-Grade Glioma (HGG) patients and 75 Low-Grade Glioma (LGG).
3. The dataset has 4 modalities- T1-weighted (T1w), post-contrast T1-weighted (T1ce), T2-weighted (T2), Fluid Attenuated Inversion Recovery (FLAIR).
4. Ground truth is also included in the dataset
5. These images are manually segmented by expert neuroradiologist labelled as as enhancing tumor (label 4), peritumoral edema (label 2), and the core (label 1)
6. Presence of Multiple tumor region is more visible with HGG than LGG- thus only HGGs are used.
7.The HGG consists of 210 patients data. The deep learning models were trained in the batches of 50 patients due to limitations of Google Colaboratory.

![image](https://user-images.githubusercontent.com/40360231/122669055-1972e280-d1d9-11eb-94b7-862e6f774ff4.png)


## 2. Pre-Processing

1. Each 3D volume of 4 has a size of 240×240×155 for all modalities and ground truth.
2. After slicing we will get 155 2D images of size 240×240 size.
3. Only 90 slices (slice no 30 to slice no 120) from each volume are selected.
4. These slices cropped to  192×192 (Background noise)
5. 18,900 2D images of each MRI modality. (210 HGG data × 90 slices = 18900)
6. Total images for all modalities will become 75,600. (210 HGG data × 90 slices × 4 modalities = 756000 images

## 3. First Approach- Sobel and Modified Unet

**EDGE FEATURE EXTRACTION USING SOBEL OPERATOR**

Image convolution along horizontal and vertical axis 
      
![image](https://user-images.githubusercontent.com/40360231/122669397-bf731c80-d1da-11eb-851c-29c526fddced.png)
      
     
_Why sobel is used?_
 
 1. Good smoothing effect to random noise
 2. Better appearance of tumor region edges due to differential of two rows and two columns
     
**MODEL DESCRIPTION**

Encoding part- performs downsampling and Decoding part-upsampling.
Down sampling is for context(tumor in this case) and up sampling is for localization(position).
Modified deep U-shaped Network (md-UNET) with multiple skipped connections.

![image](https://user-images.githubusercontent.com/40360231/122669332-74590980-d1da-11eb-8794-27c2243056a2.png)


**ENCODER PART**

-Five blocks of consecutive convolutional layers.
-Each block has two 3×3 convolutional layers with same padding and activation function as ReLU (Rectified Linear Unit), each followed by a batch normalization layer and a
 maximum pooling layer.
-A dropout layer with dropout rate of 20% is placed after two blocks to avoid overfitting.
-6th block acts as a bridge between the encoding and the decoding part, which consists of a 3×3 convolutional layer with same padding and ReLU activation function

**DECODER PART**

-Decoder part too has 5 consecutive blocks.
-Each block has a convolutional transpose layer of 3×3 with same padding and ReLU activation function merged with skip connection from previous encoder part, followed by a
 convolution layer 3×3 with same padding and ReLU activation
-Dropout Layer present after two blocks.
-Output layer with softmax activation function used for classification.

**SUMMARIZING THE MODEL**

1. No of activation filters used are 32, 64, 128, 256, 512 in five consecutive blocks of both encoder and decoder.
2. 1024 activation kernels at the center.
3. 10 Blocks
4. 21 Convolutional layers
5. 10 layers of batch normalization 
6. 5 layers of maximum pooling and there is finally one convolutional layer at the bridge point of U-shaped network.
7. Optimizer used- ADAM.
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
	  	**Dice Coefficient = (2 × | GT ∩ SEG | ) / (GT2 + SEG2 + ε)**
Where, 	GT is the standard ground truth for brain tumor,
	SEG is the predicted segmented tumorous region and ε=1e-6.
	LDice = 1 - Dice Coefficient

## 7. Experimentation and Results

1. HGG consists of data of 210 patients. The models were trained in batches of 50 patients.
2. Data split into 3:1:1- 60% images for training, 20% images for testing, and 20% for validation.
3. All the models are trained with batch size-8 and no of epochs-30

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
 
### GUI

1) Download any HGG patient's data (4 modalities' MRI scan) from BraTS 2018 dataset or from the [link](https://drive.google.com/drive/folders/1sVqDB9aXVr87UIM9g7YktPJxzKOMSDA7?usp=sharing). 
2) Run the [Anvil server IPYNB notebook](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/BTS_anvil_server.ipynb) in Google Colaboratory as it acts as a server for this anvil project. If any error occurs do factory reset runtime and run again. Ensure that all cells are executed without any errors before going to next step. 
3) Go to [link](https://bts-seg.anvil.app/) and upload the Flair, T1, T1-ce, T2 3D nifti (.nii.gz) files. Enter slice no. as the models are trained on 2D images and predictions are carried out for that particular slice only. The slice no. is expected to be in between 30 and 119 (both 2 included). Submit the data and wait for sometime as prediction is going on in the backend. 
4) You can view Flair, T1, T1ce, T2 slices of that particular slice no. entered and also the prediction of tumor regions using U-Net, Sobel Operator and modified U-Net model, V-Net, W-Net models. 

### Project Files - [Sobel and Modified U-Net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/Modified_U_Net_with_sobel_operator.ipynb), [V-Net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/2D_VNET.ipynb) and [W-net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/W_net_resblock.ipynb)

1. It is required to open the above IPYNB files in [Google Colaboratory](https://colab.research.google.com/)
2. The [link](https://drive.google.com/drive/u/2/folders/1sVqDB9aXVr87UIM9g7YktPJxzKOMSDA7) has data of first 50 HGG patients. You need to add this folder as a shortcut in Google drive.
3. Mount the google drive if asked. Add the correct path to dataset.
4. Run the IPYNB file.


## 11. Footnotes

1. You can find all batches of BraTS 2018 HGG data in the [link](https://drive.google.com/drive/folders/1_97pAToVAow2BOaPFvB2qHGCAoyB41m6?usp=sharing) and LGG data in the [link](https://drive.google.com/drive/folders/1l_H6nlGtXHDx4ie5_yg7mpx7gYMdQgfR?usp=sharing).
2. The saved models trained on first 50 patients HGG data will be found in the [link](https://drive.google.com/drive/folders/1hst3tiH6nk4IexlUiL32eVHvnb_MR7Z7?usp=sharing).

## 12. Contributors

[Ashwin Mohan](mailto:mohanashwin999@gmail.com), [Prajakta Bhosale](mailto:prajatabhosale3333@gmail.com), [Prajakta Joshi](mailto:prajakta.joshi1999@gmail.com), [Ashley George](mailto:ashleygeorge1999@gmail.com) - Pimpri Chinchwad College of Engineering, Pune, Maharashtra India.

**Guided by:** [Prof. Sonal Gore](mailto:sonalgore@gmail.com) - Pimpri Chinchwad College of Engineering, Pune, Maharashtra, India.
