import logging
import os,random
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch,monai

from monai import config
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader,list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,MeanIoU,HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Spacingd,
    Activationsd, 
    LoadImaged, 
    AsDiscreted, 
    Compose, 
    SaveImaged, 
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,Invertd,
    )
from Vnet_train_val import CTNormalizationd

def make_deterministic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
make_deterministic(42)


def main(tempdir):

    images = sorted(glob(os.path.join(tempdir, 'train/image',"*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, 'train/label',"*.nii.gz")))

    val_images = sorted(glob(os.path.join(tempdir, 'val/image',"*.nii.gz")))
    val_segs = sorted(glob(os.path.join(tempdir, 'val/label',"*.nii.gz")))
    
    test_images = sorted(glob(os.path.join(tempdir, 'test/image',"*.nii.gz")))
    test_segs = sorted(glob(os.path.join(tempdir, 'test/label',"*.nii.gz")))
    
    test_files = [{"image": img, "label": seg} for img, seg in zip(test_images, test_segs)]
    
    train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.4882810115814209,0.4882810115814209,4.649400138854981),
            mode=("bilinear", "nearest"),
        ),
        CTNormalizationd(keys=['image'],intensity_properties={'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),
        RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(128,128,32), pos=1,neg=1,num_samples=4,image_key='image',image_threshold=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ]
    )
    val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.4882810115814209,0.4882810115814209,4.649400138854981),
            mode=("bilinear", "nearest"),
        ),
        CTNormalizationd(keys=['image'],intensity_properties={'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),
    ]
    )
    
    post_transform = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=val_transform,
                orig_keys="label",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            SaveImaged(keys="pred", output_dir="./SwinUNETR_out", output_ext=".nii.gz", output_postfix="seg"),

        ]
    )
    
    test_ds = monai.data.Dataset(data=test_files, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    test_check_data = monai.utils.misc.first(test_loader)
    print(test_check_data["image"].shape, test_check_data["label"].shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.SwinUNETR(
        img_size=(128,128,32),
        in_channels=1,
        out_channels=1, 
        feature_size=48,
    ).to(device)
    model.load_state_dict(torch.load("SwinUNETR_best_metric_model_segmentation3d_array.pth"))
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric =  MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95, reduction="mean", get_not_nans=False)
    
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data['image'].to(device), test_data['label'].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (128,128,32)
            sw_batch_size = 4
            test_data['pred'] = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
            test_data = [post_transform(i) for i in decollate_batch(test_data)]

  


if __name__ == "__main__":
    temdir = '/home/user416/songcw/data/IPH_IVH/IPH_IVH'
    main(temdir)