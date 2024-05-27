# HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans
This repository contains material associated to this  ***[paper](https://arxiv.org/pdf/2405.14559)***


It contains:

  - The acquisition and information of this dataset.

  - The code and description of the baseline for this dataset.

If you use this material, we would appreciate if you could cite the following reference.
## Citation
```text
@inproceedings{
  Song2024HemSeg200AV,
  title={HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans},
  author={Changwei Song and Qing Zhao and Jianqiang Li and Xin Yue and Ruoyun Gao and Zhaoxuan Wang and An Gao and Guanghui Fu},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:269982589}
}
```

## Datasets
- The format is .nii.gz
- Image:

  The original data comes from the [RSNA Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data), which includes image level annotations for cerebral hemorrhage types but does not provide pixel level annotations.

  However, this dataset only provides 2D data of the Dicom type. We convert to 3D format data of type nii.gz through the following process.

  ```bash
  pip install SimpleITK
  ```

  ```python
  import SimpleITK as sitk
  import os

  def convert_dicom_to_nifti(dicom_directory, output_file):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    sitk.WriteImage(image, output_file)

  dicom_directory = 'path_to_dicom_directory'
  output_file = 'output_file.nii.gz'
  convert_dicom_to_nifti(dicom_directory, output_file)
  ```
  Set `path_to_dicom_directory` as the folder path containing DICOM files, and set `output_file.nii.gz` as the path to the saved NIfTI file

- Annotation:

  The segmentation labels of this dataset are labeled by us and can be obtained from the following link: https://pan.baidu.com/s/1b_GR3hE1rIr6HHKUAXqftA?pwd=q02q


## Baseline

We evaluated seven commonly used 3D medical image segmentation models in the field, which can facilitate the understanding of the performance of these commonly used algorithms on this dataset.

Result: The performance of IPH and IVH segmentation on brain CT scans from RSNA dataset. Results presented as mean with 95% bootstrap confidence interval computed on the independent test set.

| Models         | Dice(%)                  | IoU(%)                  | HD95                     |
|----------------|--------------------------|-------------------------|--------------------------|
| U-net[1]          | 63.70 [55.82, 71.10]     | 51.40 [44.26, 58.48]    | 83.48 [51.66, 88.37]     |
| V-net[2]          | 61.19 [52.81, 69.34]     | 49.39 [41.79, 56.98]    | 48.88 [33.30, 66.18]     |
| SegResNet[3]      | 32.66 [25.96, 39.74]     | 22.02 [16.99, 27.57]    | 141.16 [124.20, 158.99]  |
| Attention U-net[4]| 66.77 [58.90, 73.86]     | 54.98 [47.42, 62.10]    | 49.62 [33.44, 67.26]     |
| UNETR[5]          | 55.98 [48.94, 62.69]     | 42.43 [36.10, 48.72]    | 111.10 [90.80, 131.54]   |
| Swin UNETR[6]     | 48.50 [40.79, 56.13]     | 35.92 [29.19, 42.59]    | 164.85 [143.83, 186.99]  |
| nnU-net[7]        | **78.99 [73.21, 83.86]** | **68.49 [62.05, 74.25]**| **22.16 [9.19, 38.20]**  |


## References

```text
[1]Çiçek Ö, Abdulkadir A, Lienkamp S S, et al. 3D U-Net: learning dense volumetric segmentation from sparse annotation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer International Publishing, 2016: 424-432.
[2]Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//2016 fourth international conference on 3D vision (3DV). Ieee, 2016: 565-571.
[3]Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization[C]//Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4. Springer International Publishing, 2019: 311-320.
[4]Oktay O, Schlemper J, Folgoc L L, et al. Attention u-net: Learning where to look for the pancreas. arxiv 2018[J]. arxiv preprint arxiv:1804.03999, 1804.
[5]Hatamizadeh A, Tang Y, Nath V, et al. Unetr: Transformers for 3d medical image segmentation[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2022: 574-584.
[6]Hatamizadeh A, Nath V, Tang Y, et al. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images[C]//International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021: 272-284.
[7]Isensee F, Jaeger P F, Kohl S A A, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation[J]. Nature methods, 2021, 18(2): 203-211.
```
