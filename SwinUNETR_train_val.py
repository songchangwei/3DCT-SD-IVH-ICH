import logging
import os
import sys
import tempfile
from glob import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader, pad_list_data_collate,list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    Spacingd,
    Orientationd,
    RandCropByPosNegLabel,
    LoadImaged,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    SpatialPadd,
)
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import Transform,MapTransform
from torch.optim.lr_scheduler import StepLR


class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        """
        初始化CTNormalization转换。
        :param keys: 字典中要转换的键列表
        :param intensity_properties: 包含强度相关属性的字典（均值、标准差、百分位数边界等）
        :param target_dtype: 转换目标的数据类型
        """
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
        """
        在图像上应用CT标准化。
        :param data: 包含图像数据的字典
        :return: 包含标准化图像数据的字典
        """
        d = dict(data)
        for key in self.keys:
            assert self.intensity_properties is not None, "CTNormalizationd requires intensity properties"
            d[key] = d[key].astype(self.target_dtype)
            mean_intensity = self.intensity_properties['mean']
            std_intensity = self.intensity_properties['std']
            lower_bound = self.intensity_properties['percentile_00_5']
            upper_bound = self.intensity_properties['percentile_99_5']
            d[key] = np.clip(d[key], lower_bound, upper_bound)
            d[key] = (d[key] - mean_intensity) / max(std_intensity, 1e-8)
        return d


def main(tempdir):

    images = sorted(glob(os.path.join(tempdir, 'train/image',"*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, 'train/label',"*.nii.gz")))

    val_images = sorted(glob(os.path.join(tempdir, 'val/image',"*.nii.gz")))
    val_segs = sorted(glob(os.path.join(tempdir, 'val/label',"*.nii.gz")))
    
    train_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]
    val_files = [{"image": img, "label": seg} for img, seg in zip(val_images, val_segs)]
    
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
        SpatialPadd(keys=["image", "label"], spatial_size=(512, 512, 32), method="symmetric"),
        RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(128,128,32), pos=1,neg=1,num_samples=2,image_key='image',image_threshold=0),
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
        #RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(256,256,16), pos=1,neg=1,num_samples=4,image_key='image',image_threshold=0,),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ]
    )
    
    
    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transform)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    
    train_check_data = monai.utils.misc.first(train_loader)
    print(train_check_data["image"].shape, train_check_data["label"].shape)
    
    val_check_data = monai.utils.misc.first(val_loader)
    print(val_check_data["image"].shape, val_check_data["label"].shape)
    
    '''

    train_imtrans = Compose(
        [
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim = (0.4882810115814209,0.4882810115814209,4.649400138854981),mode='bilinear'),
            RandCropByPosNegLabel(spatial_size=(256,256,16), pos=1,neg=1,num_samples=4),
            CTNormalization({'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    train_segtrans = Compose(
        [
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim = (0.4882810115814209,0.4882810115814209,4.649400138854981),mode='nearest'),
            RandCropByPosNegLabel(spatial_size=(256,256,16), pos=1,neg=1,num_samples=4),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    val_imtrans = Compose(
        [
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim = (0.4882810115814209,0.4882810115814209,4.649400138854981),mode='bilinear'),
            CTNormalization({'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),

        ]
    
    )
    val_segtrans = Compose(
        [
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim = (0.4882810115814209,0.4882810115814209,4.649400138854981),mode='nearest'),

        ]
        
    )
    
    #   define image dataset, data loader
    train_ds = ImageDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=16, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)
    val_check_ds = ImageDataset(val_images, val_segs, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_check_ds, batch_size=16, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)
    im, seg = monai.utils.misc.first(train_loader)
    print(im.shape, seg.shape)
    val_im, val_seg = monai.utils.misc.first(val_loader)
    print(val_im.shape, val_seg.shape)
    '''

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.SwinUNETR(
        img_size=(128,128,32),
        in_channels=1,
        out_channels=1, 
        feature_size=48,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)


    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter('runs/SwinUNETR')
    for epoch in range(200):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{200}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        #更新学习率
        scheduler.step()
        # 打印当前学习率
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1}/{200}, Current Learning Rate: {current_lr}")
        

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_epoch_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    roi_size = (128,128,32)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    #val_epoch_len = len(val_check_ds)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_epoch_loss += val_loss.item()
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                print("current epoch: {} current val_loss: {:.4f}".format(epoch + 1, val_epoch_loss/val_step))
                writer.add_scalar("val_loss", val_epoch_loss/val_step, epoch + 1)
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "SwinUNETR_best_metric_model_segmentation3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                #plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                #plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                #plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    

if __name__ == "__main__":
    temdir = '/home/user416/songcw/data/IPH_IVH/IPH_IVH'
    main(temdir)