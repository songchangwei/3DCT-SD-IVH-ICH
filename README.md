# HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans
This repository contains material associated to this  ***[paper](https://arxiv.org/pdf/2405.14559)*** 


It contains:

  - The acquisition and information of this dataset.

  - The code and description of the baseline for this dataset.

If you use this material, we would appreciate if you could cite the following reference.
## Citation
@inproceedings{

  Song2024HemSeg200AV,

  title={HemSeg-200: A Voxel-Annotated Dataset for 
  Intracerebral Hemorrhages Segmentation in Brain CT Scans},

  author={Changwei Song and Qing Zhao and Jianqiang Li and Xin Yue and Ruoyun Gao and Zhaoxuan Wang and An Gao and Guanghui Fu},

  year={2024},

  url={https://api.semanticscholar.org/CorpusID:269982589}

}

## Datasets
- The format is .nii.gz
- image:
  
  The original data comes from the [RSNA Cerebral Hemorrhage Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data), which includes image level annotations for cerebral hemorrhage types but does not provide pixel level annotations.

  However, this dataset only provides 2D data of the Dicom type. We convert to 3D format data of type nii.gz through the following process.

  ```bash
  pip install SimpleITK
  ```



  ```python
  import SimpleITK as sitk
  import os

  def convert_dicom_to_nifti(dicom_directory, output_file):
    # 读取DICOM系列
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    # 将图像写入NIfTI文件
    sitk.WriteImage(image, output_file)
  ```
  Set `path_to_dicom_directory` as the folder path containing DICOM files, and set `output_file.nii.gz` as the path to the saved NIfTI file

- annotation:
  
  The segmentation labels of this dataset are labeled by us and can be obtained from the following link: https://pan.baidu.com/s/1b_GR3hE1rIr6HHKUAXqftA?pwd=q02q 
