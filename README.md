# Brain-Tumor-Segmentation-Using-Deep-Neural-Networks

## Description
This project presents the use of deep learning and image processing techniques for the segmentation of tumors into different region. We used the following three approaches for segmentation of glioma brain tumor.
1. The first approach is a hybrid system within which we use Sobel operator which is an Image Processing technique for edge clarification and a modified U-Net which is a Deep Learning Neural Network for training. 
2. The second approach proposes a V-Net model architecture for tumor segmentation.
3. The third approach proposes a W-Net model architecture for tumor segmentation.

## 1. Dataset

1. Data released with multimodal brain tumor segmentation challenge by Medical Image Computing and Computer Assisted Intervention (MICCAI) is used for training.
2. 3D dataset consists of pre surgical MRI scans of 210 High-Grade Glioma (HGG) patients and 75 Low-Grade Glioma (LGG).
3. The dataset has 4 modalities- T1-weighted (T1w), post-contrast T1-weighted (T1ce), T2-weighted (T2), Fluid Attenuated Inversion Recovery (FLAIR).
4. Ground truth is also included in the dataset
5. These images are manually segmented by expert neuroradiologist labelled as as enhancing tumor (label 4), peritumoral edema (label 2), and the core (label 1)
6. Presence of Multiple tumor region is more visible with HGG than LGG- thus only HGGs are used.
7. The HGG consists of 210 patients data. The deep learning models were trained in the batches of 50 patients due to limitations of Google Colaboratory.

## 2. Pre-Processing

1. Each 3D volume of 4 has a size of 240×240×155 for all modalities and ground truth.
2. After slicing we will get 155 2D images of size 240×240 size.
3. Only 90 slices (slice no 30 to slice no 120) from each volume are selected.
4. These slices cropped to  192×192 (Background noise)
5. 18,900 2D images of each MRI modality. (210 HGG data × 90 slices = 18900)
6. Total images for all modalities will become 75,600. (210 HGG patient's data × 90 slices of each modality × 4 modalities = 75,600 2D images)

## 3. Training

The models were trained using 3 approaches:
1. The first approach is a hybrid system within which we use Sobel operator which is an Image Processing technique for edge clarification and a modified U-Net which is a Deep Learning Neural Network for training. 
2. The second approach proposes V-Net model architecture for tumor segmentation.
3. The third approach proposes W-Net model architecture for tumor segmentation.

## 4. Experimentation

1. HGG consists of data of 210 patients. The models were trained in batches of 50 patients.
2. Data split into 3:1:1- 60% images for training, 20% images for testing, and 20% for validation.
3. All the models are trained with batch size-8 and no of epochs-30.

## 5. Conclusion

1. Several Deep Learning and Image processing techniques were studied in the due course of the project. 
2. We completed research work with current usage of techniques in image preprocessing, image segmentation, common feature extraction and classification recently used were
   analyzed and studied. We choose in total 3 systems-
      
      i. SOBEL + MD-UNET (Dice score-0.9918)
      ii. 2D -VNET (Dice score-0.9947)
      iii. **WNET (Dice score-0.9964)**
3. The output we received is of tumor region highlighted into three regions which are edema, enhancing tumor and non enhancing tumor. We achieved the **highest dice score of
    99.64% with WNET** architecture on training the dataset through our system.

 
## 6. GUI of Project

![image](https://user-images.githubusercontent.com/40360231/122669670-12010880-d1dc-11eb-8845-bd2b2aef5b00.png)

Link to website: https://bts-seg.anvil.app/

## 7. How to Run?
 
### GUI

1. Download any HGG patient's data (4 modalities' MRI scan) from BraTS 2018 dataset or from the [link](https://drive.google.com/drive/folders/1sVqDB9aXVr87UIM9g7YktPJxzKOMSDA7?usp=sharing). 
2. Run the [Anvil server IPYNB notebook](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/BTS_anvil_server.ipynb) in Google Colaboratory as it acts as a server for this anvil project. If any error occurs do factory reset runtime and run again. Ensure that all cells are executed without any errors before going to next step. 
3. Go to [link](https://bts-seg.anvil.app/) and upload the Flair, T1, T1-ce, T2 3D nifti (.nii.gz) files. Enter slice no. as the models are trained on 2D images and predictions are carried out for that particular slice only. The slice no. is expected to be in between 30 and 119 (both 2 included). Submit the data and wait for sometime as prediction is going on in the backend. 
4. You can view Flair, T1, T1ce, T2 slices of that particular slice no. entered and also the prediction of tumor regions using U-Net, Sobel Operator and modified U-Net model, V-Net, W-Net models. 

### Project Files - [Sobel and Modified U-Net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/Modified_U_Net_with_sobel_operator.ipynb), [V-Net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/2D_VNET.ipynb) and [W-net](https://github.com/Brain-Tumor-Segmentation/brain-tumor-segmentation-using-deep-neural-networks/blob/main/Codes/W_net_resblock.ipynb)

1. It is required to open the above IPYNB files in [Google Colaboratory](https://colab.research.google.com/)
2. The [link](https://drive.google.com/drive/u/2/folders/1sVqDB9aXVr87UIM9g7YktPJxzKOMSDA7) has data of first 50 HGG patients. You need to add this folder as a shortcut in Google drive.
3. Mount the google drive if asked. Add the correct path to dataset.
4. Run the IPYNB file.


## 8. Footnotes

1. You can find all batches of BraTS 2018 HGG data in the [link](https://drive.google.com/drive/folders/1_97pAToVAow2BOaPFvB2qHGCAoyB41m6?usp=sharing) and LGG data in the [link](https://drive.google.com/drive/folders/1l_H6nlGtXHDx4ie5_yg7mpx7gYMdQgfR?usp=sharing).
2. The saved models trained on first 50 patients HGG data will be found in the [link](https://drive.google.com/drive/folders/1hst3tiH6nk4IexlUiL32eVHvnb_MR7Z7?usp=sharing).

## 9. Contributors

[Sonal Gore](mailto:sonalgore@gmail.com), [Ashwin Mohan](mailto:mohanashwin999@gmail.com), [Prajakta Bhosale](mailto:prajatabhosale3333@gmail.com), [Prajakta Joshi](mailto:prajakta.joshi1999@gmail.com), [Ashley George](mailto:ashleygeorge1999@gmail.com) - Pimpri Chinchwad College of Engineering, Pune, Maharashtra India.

[Dr. Jayant Jagtap](mailto:jayantjagtap87@gmail.com) - Symbiosis Institute of Technology, Pune, Maharashtra, India.
