import logging
import os
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
from monai.transforms import Spacingd,Activations, LoadImaged, AsDiscrete, Compose, SaveImage, EnsureTyped,EnsureChannelFirstd,Orientationd
from Vnet_train_val import CTNormalizationd

def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = sorted(glob(os.path.join(tempdir, 'test/image',"*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, 'test/label',"*.nii.gz")))
    val_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]


    # define transforms for image and segmentation
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
        #RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(256,256,16), pos=1,neg=1,num_samples=4,image_key='image',image_threshold=0,),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    val_check_data = monai.utils.misc.first(val_loader)
    print(val_check_data["image"].shape, val_check_data["label"].shape)
    
    
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    iou_metric =  MeanIoU(include_background=True, reduction="none", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95, reduction="none", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir="./SwinUNETR", output_ext=".nii.gz", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.SwinUNETR(
        img_size=(128,128,32),
        in_channels=1,
        out_channels=1, 
        feature_size=48,
    ).to(device)

    model.load_state_dict(torch.load("SwinUNETR_best_metric_model_segmentation3d_array.pth"))
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (128,128,32)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            iou_metric(y_pred=val_outputs, y=val_labels)
            hd_metric(y_pred=val_outputs, y=val_labels)
            #for val_output in val_outputs:
            #    saver(val_output)
        # aggregate the final mean dice result
        print("evaluation dice metric:", dice_metric.aggregate(reduction ='mean').item())
        print("evaluation iou metric:", iou_metric.aggregate(reduction ='mean').item())
        print("evaluation hd metric:", hd_metric.aggregate(reduction ='mean').item())
        # reset the status
        #print(type(dice_metric))
        #print(dice_metric.aggregate().detach().cpu().numpy())
        np.save('SwinUNTER_dice_score.npy', dice_metric.aggregate().detach().cpu().numpy())
        #print(iou_metric.aggregate().detach().cpu().numpy())
        np.save('SwinUNTER__iou_metric.npy', iou_metric.aggregate().detach().cpu().numpy())
        #print(hd_metric.aggregate().detach().cpu().numpy())
        np.save('SwinUNTER__hd_metric.npy', hd_metric.aggregate().detach().cpu().numpy())
        dice_metric.reset()
        iou_metric.reset()
        hd_metric.reset()


if __name__ == "__main__":
    temdir = '/home/user416/songcw/data/IPH_IVH/IPH_IVH'
    main(temdir)
