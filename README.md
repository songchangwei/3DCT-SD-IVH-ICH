# HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans
This repository contains material associated to this  ***[paper](https://arxiv.org/pdf/2405.14559)***


It contains:

  - Link to the dataset: [CT images](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data) from RSNA challenge [2] and [label file](https://pan.baidu.com/s/1b_GR3hE1rIr6HHKUAXqftA?pwd=q02q) that we annotated.
  - The [code](#Data-selection-and-conversion) to select and cover DICOM file to NIFTI file that we annotated.
  - The [code](#Baseline-model-implementations) for the baseline models that we implemented.

If you use this material, we would appreciate if you could cite the following reference.
## Citation
* The paper [1] that contains the annotated dataset: 
  ```text
  @article{song2024hemseg,
    title={HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans},
    author={Song, Changwei and Zhao, Qing and Li, Jianqiang and Yue, Xin and Gao, Ruoyun and Wang, Zhaoxuan and Gao, An and Fu, Guanghui},
    journal={arXiv preprint arXiv:2405.14559},
    year={2024}
  }
  ```

* The paper of the [RSNA challenge dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
  ```text
  @article{flanders2020construction,
    title={Construction of a machine learning dataset through collaboration: the RSNA 2019 brain CT hemorrhage challenge},
    author={Flanders, Adam E and Prevedello, Luciano M and Shih, George and Halabi, Safwan S and Kalpathy-Cramer, Jayashree and Ball, Robyn and Mongan, John T and Stein, Anouk and Kitamura, Felipe C and Lungren, Matthew P and others},
    journal={Radiology: Artificial Intelligence},
    volume={2},
    number={3},
    pages={e190211},
    year={2020},
    publisher={Radiological Society of North America}
  }
  ```

* The link of the [RSNA challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
  ```text
  @misc{rsna_kaggle,
      author = {Anouk, Stein MD and Carol, Wu and Chris, Carr and George, Shih and Jayashree, Kalpathy-Cramer and Julia, Elliott kalpathy and Luciano, Prevedello and Marc, Kohli MD and Matt, Lungren and Phil, Culliton and Robyn, Ball and Safwan, Halabi MD},
      title = {RSNA Intracranial Hemorrhage Detection},
      publisher = {Kaggle},
      year = {2019},
      url = {https://kaggle.com/competitions/rsna-intracranial-hemorrhage-detection}
  }
  ```
## Packages install
``` python3
jsonlines==4.0.0
nibabel==5.2.1
numpy==2.0.0
pandas==2.0.3
pydicom==2.4.4
SimpleITK==2.3.1
```
## Datasets
### Data selection and conversion
The CT images are from the [RSNA Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data). However, due to the copyright restrictions of this challenge, we cannot provide the DICOM or the converted NIFTI files directly. We encourage you to join and download the RSNA challenge dataset from the [Kaggle platform](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data). You can then use our code to select and convert the DICOM files into NIFTI files, which we have annotated. These files can be used for the hemorrhage segmentation task that we proposed. It contains these following step:

1. Download the original CT images from the [RSNA Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data): https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data.
2. Download the [label files](https://github.com/songchangwei/3DCT-SD-IVH-ICH/blob/main/data/IPH_IVH_label.zip) that we annotated: [https://pan.baidu.com/s/1b_GR3hE1rIr6HHKUAXqftA?pwd=q02q](https://github.com/songchangwei/3DCT-SD-IVH-ICH/blob/main/data/IPH_IVH_label.zip)
3. The [`annotion_file_info.jsonl`](https://github.com/songchangwei/3DCT-SD-IVH-ICH/blob/main/annotion_file_info.jsonl) file that we selected to annotate. It contains two subtype of brain hemorrages: IPH and IVH. 
4. Utilize [`dcm2nii.py`](https://github.com/songchangwei/3DCT-SD-IVH-ICH/blob/main/dcm2nii.py) to select the selected DICOM file, and convert them into NIfTI format. You need to provide two parameters: the path to the raw data file from Kaggle and the path for the NIfTI files to be saved. Usage is as follows:


   ```python3
   # Example code here
   python dcm2nii.py path_to_input_dir path_to_output_dir
   ```


## Baseline model implementations and evaluations
The models we implemented are source from [MONAI package](https://monai.io/) [10]. We evaluated seven commonly used 3D medical image segmentation models in the field, which helps to understand the performance of these commonly used algorithms on this dataset. It contains: 
* The codes related to model training can be found at: [`model_train`]() folder
* The codes related to model inferencing can be found at: [`model_inference`]() folder
* The codes about model evaluation can be found at [`evaluation`]() folder.
    * ``evaluation/eval_bootstrap_ci.py``: This code is for evaluation and calculate the 95% bootstrap confidence interval.

The performance of the experimental models can be seen in **Table 1**.  

**Table 1:** The performance of IPH and IVH segmentation on brain CT scans from RSNA dataset. Results presented as mean with 95% bootstrap confidence interval computed on the independent test set.

| Models             | Dice(%)                  | IoU(%)                   | HD95                    |
| ------------------ | ------------------------ | ------------------------ | ----------------------- |
| U-net[3]           | 63.70 [55.82, 71.10]     | 51.40 [44.26, 58.48]     | 83.48 [51.66, 88.37]    |
| V-net[4]           | 61.19 [52.81, 69.34]     | 49.39 [41.79, 56.98]     | 48.88 [33.30, 66.18]    |
| SegResNet[5]       | 32.66 [25.96, 39.74]     | 22.02 [16.99, 27.57]     | 141.16 [124.20, 158.99] |
| Attention U-net[6] | 66.77 [58.90, 73.86]     | 54.98 [47.42, 62.10]     | 49.62 [33.44, 67.26]    |
| UNETR[7]           | 55.98 [48.94, 62.69]     | 42.43 [36.10, 48.72]     | 111.10 [90.80, 131.54]  |
| Swin UNETR[8]      | 48.50 [40.79, 56.13]     | 35.92 [29.19, 42.59]     | 164.85 [143.83, 186.99] |
| nnU-net[9]         | **78.99 [73.21, 83.86]** | **68.49 [62.05, 74.25]** | **22.16 [9.19, 38.20]** |


## References

1. Song, Changwei, et al. "HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans." arXiv preprint arXiv:2405.14559 (2024).
2. Flanders, Adam E., et al. "Construction of a machine learning dataset through collaboration: the RSNA 2019 brain CT hemorrhage challenge." Radiology: Artificial Intelligence 2.3 (2020): e190211.
3. Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing, 2016: 424-432.
4. Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//2016 fourth international conference on 3D vision (3DV). Ieee, 2016: 565-571.
5. Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization[C]//Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4. Springer International Publishing, 2019: 311-320.
6. Oktay O, Schlemper J, Folgoc L L, et al. Attention u-net: Learning where to look for the pancreas. arxiv 2018[J]. arxiv preprint arxiv:1804.03999, 1804.
7. Hatamizadeh A, Tang Y, Nath V, et al. Unetr: Transformers for 3d medical image segmentation[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2022: 574-584.
8. Hatamizadeh A, Nath V, Tang Y, et al. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images[C]//International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021: 272-284.
9. Isensee F, Jaeger P F, Kohl S A A, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation[J]. Nature methods, 2021, 18(2): 203-211.
10. Cardoso, M. Jorge, et al. "MONAI: An open-source framework for deep learning in healthcare." arXiv preprint arXiv:2211.02701 (2022).

