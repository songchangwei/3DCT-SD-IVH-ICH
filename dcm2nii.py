import SimpleITK as sitk
import os
import sys,pydicom
import nibabel as nb
import numpy as np
'''
功能：读取filepath下的dcm文件
返回值：读取得到的SimpleITK.SimpleITK.Image类   
其他说明：  file = sitk.ReadImage(filepath)
            获取基本信息，大小，像素间距，坐标原点，方向
            file.GetSize()
            file.GetOrigin()
            file.GetSpacing()
            file.GetDirection()
'''



def readdcm(filepath,target_filepath):
   #filepath = "./T2"
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader() #读取数据端口
    series_reader.SetFileNames(series_file_names)  #读取名称
    images = series_reader.Execute()#读取数据
    #print(images.GetSpacing())
    sitk.WriteImage(images, target_filepath)#保存为nii
    #return images

def dcm2nii(filepath,target_filepath):
    sys.path.append(r'C:\Users\Administrator\Downloads\NITRC-multi-file-downloads\MRIcroGL_windows\MRIcroGL\Resources')
    os.system('chcp 65001')
    os.system(r'C:\Users\Administrator\Desktop\MRIcroGL_windows\MRIcroGL\Resources\dcm2niix {0}'.format(filepath))


def readfile(source_dir,target_dir):
    lesion_names = ['label_1','label_2']
    for lesion in lesion_names:
        patient_names = os.listdir(source_dir+'/'+lesion)
        for patient in patient_names:
            scan_names = os.listdir(source_dir+'/'+lesion+'/'+patient)
            for scan in scan_names:
                source_file = source_dir+'/'+lesion+'/'+patient+'/'+scan
                target_file = target_dir+'/'+lesion+'/'+patient+'/'+scan+'.nii.gz'
                print(source_file,target_file)
                if not os.path.exists(target_dir+'/'+lesion+'/'+patient):
                    os.mkdir(target_dir+'/'+lesion+'/'+patient)
                readdcm(source_file,target_file)
 
 
if __name__ == '__main__':
    filepath =  "F:\\barin_chuxue_dcm"  #dcm文件保存路径
    target_filepath = 'F:\\brain_chuxue_nii' #nii文件保存路径
    readfile(filepath,target_filepath)