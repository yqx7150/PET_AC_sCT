# PET_AC_sCT
Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction


# Abstract
Attenuation correction (AC) is essential for the generation of artifact-free and quantitatively accurate posi-tron emission tomography (PET) images. However, AC of PET faces challenges including inter-scan motion, image artifacts such as truncation or distortion, and erroneous transformation of structural voxel-intensities to PET attenuation-correction factors. Nowadays, the problem of AC for quantitative PET had been solved to a large extent after the commercial availability of devices combining PET with computed tomography (CT). Meanwhile, considering the feasibility of a deep learning approach for PET AC without anatomical imaging, this paper develops a PET AC method, which uses deep learning to generate continuously valued CT images from non-attenuation corrected PET images for AC on brain PET imaging. Specifically, an invertible network combined with a variable augmentation strategy that can achieve the bidirectional inference processes is pro-posed for synthetic CT generation (IVNAC). In addition, data collection for training and evaluation of the pro-posed method is performed utilizing an all-digital PET, which further guarantees the high quality of the NAC PET dataset compared to traditional PET. Perceptual analysis and quantitative evaluations illustrate that the invertible network for PET AC outperforms other existing AC models. Furthermore, with the proposed method shows great similarity to reference CT images both qualitatively and quantitatively, which demonstrates great potential for brain PET AC in the absence of structural information.


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

### Other Related Projects

  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)        

  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
   
