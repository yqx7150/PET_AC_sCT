# PET_AC_sCT

**Paper**:   Synthetic CT Generation via Variant Invertible Network for Brain PET Attenuation Correction 

**Authors**:   Yu Guan, Bohui Shen, Shirui Jiang, Xinchong Shi, Xiangsong Zhang, Bingxuan Li, Qiegen Liu*       
 
IEEE Transactions on Radiation and Plasma Medical Sciences    

https://ieeexplore.ieee.org/document/10666843  

Date : 9-September-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, School of Mathematics and Computer Sciences, Nanchang University.  

# Abstract
Attenuation correction (AC) is essential for the generation of artifact-free and quan-titatively accurate positron emission tomography (PET) images. Nowadays, deep-learning-based methods have been extensively applied to PET AC tasks, yielding promising results. Therefore, this paper develops an innovative approach to generate continuously valued CT images from non-attenuation corrected PET images for AC on brain PET imaging. Specifically, an invertible neural network combined with the variable augmentation strategy that can achieve the bidirectional inference processes is proposed for synthetic CT generation. On the one hand, invertible architecture ensures a bijective mapping between the PET and synthetic CT image spaces, which can potentially improve the robustness of the prediction and provide a way to validate the synthetic CT by checking the consistency of the inverse mapping. On the other hand, the variable augmentation strategy enriches the training process and leverages the intrinsic data properties more effectively. Therefore, the combination provides for superior performance in PET AC by preserving information throughout the network and by better handling of the data variability inherent PET AC. To evaluate the performance of the proposed algorithm, we conducted a comprehensive study on a total of 1480 2D slices from 37 whole-body 18F-FDG clinical patients using comparative algorithms (such as Cycle-GAN and Pix2pix). Perceptual analysis and quantitative evaluations illustrate that the invertible network for PET AC outperforms other existing AC models, which demonstrates the feasibility of achieving brain PET AC without additional anatomical information.


## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/figure 1.png" width = "1000" height = "520">  </div>
 **The schematic flow diagram of the proposed method. The training phase is first performed with NAC PET and reference CT images, after which the well-trained network is fixed and ready for generating synthetic CT images for new PET data in the reconstruction phase.**
 

<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/Fig 3.png" width = "1000" height = "520"> </div>
 The pipeline of IVNAC. Invertible model is composed of both forward and inverse process. We illustrate the details of the invertible block on the bottom. s , t and r are transformations defined in the bijective functions.


## Training Demo
```bash
python train_lr.py --task=1to1 --out_path="./results/new_exp/" --root2='./data_for_training/pet_mat' --root3='./data_for_training/pet_mat' --root1='./data_for_training/ct_mat'
```
## Test Demo
```bash
python test.py --task=1to1 --out_path="./results/exp/" --root2='./data_for_test/pet_mat' --root3='./data_for_test/pet_mat' --root1='./data_for_test/ct_mat' --ckpt="./results/exp/1to1/checkpoint/0028.pth"
```
## Checkpoints
We provide a pretrained checkpoint. You can run the above command to use the pretrained model directly


## Synthetic CT Generation and Evaluation
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/figure 5.png" width = "1000" height = "1200"> </div>
 Examples of synthetic CT image on a patient’s brain. Five columns from left to right are NAC-PET, reference-CT, Cycle-GAN-CT, Pix2pix-CT and IVNAC-CT, respectively. The second row shows the difference images between the reference CT and the synthetic CT.


## Synthetic CT to PET Attenuation Correction
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/figure 6.png" width = "1000" height = "1200"> </div>
 PET data reconstructed with reference and generated synthesized CT images alongside their voxel-wise difference map. Five columns from left to right are reference-CT, AC-PET, Cycle-GAN-PET, Pix2pix-PET and IVNAC-PET, respectively.


<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/figure 7.png" width = "1000" height = "450"> </div>
 Unique example of PET data reconstructed with reference and generated synthesized CT images alongside their voxel-wise difference map. Red arrows indicate region with relatively obvious details and textures and the intensity of residual maps is two times magnified.


### Other Related Projects

  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

 * Temporal Image Sequence Separation in Dual-Tracer Dynamic PET With an Invertible Network  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN)        
       
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)        

  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       
    
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
   
