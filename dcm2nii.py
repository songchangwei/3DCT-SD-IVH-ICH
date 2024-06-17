import SimpleITK as sitk
import os
import sys,pydicom
import nibabel as nb
import numpy as np
import jsonlines,json
import os,shutil
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


 
if __name__ == '__main__':
    filepath =  "path_to_input_dir"  #the downloaded raw files are stored in the specified path.
    target_filepath = 'path_to_output_dir' #the NIfTI files are saved in the designated path.
    with jsonlines.open('annotion_file_info.jsonl','r') as reader:
        for item in jsonlines.Reader(reader):
            scan_name = item['scan']
            slice_list = item['slice_list']
            os.mkdir(scan_name) #创建临时文件
            for slice in slice_list:
                source_file = os.path.join(filepath,slice+'dcm')
                target_file = os.path.join(scan_name,slice+'dcm')
                shutil.copyfile(source_file,target_file)
            readdcm(scan_name,os.path.join(target_filepath,scan_name+'.nii.gz'))

            
