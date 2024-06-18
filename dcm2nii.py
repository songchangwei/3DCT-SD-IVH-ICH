import SimpleITK as sitk
import os
import sys,pydicom
import nibabel as nb
import numpy as np
import jsonlines,json
import os,shutil
import sys




def dcm2nii(filepath,target_filepath):
   #filepath = "./T2"
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader() 
    series_reader.SetFileNames(series_file_names)  
    images = series_reader.Execute()
    #print(images.GetSpacing())
    sitk.WriteImage(images, target_filepath)
    #return images




 
if __name__ == '__main__':
    filepath = sys.argv[1]
    target_filepath = sys.argv[2]
    #filepath =  "rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train"  #the downloaded raw files are stored in the specified path.
    #target_filepath = 'data' #the NIfTI files are saved in the designated path.
    with jsonlines.open('annotion_file_info.jsonl','r') as reader:
        for item in reader:
            scan_name = item['scan_']
            slice_list = item['slice_list']
            if not os.path.exists(scan_name):
                os.mkdir(scan_name) #A temporary file has been created.
            for slice in slice_list:
                source_file = os.path.join(filepath,'ID_'+slice+'.dcm')
                target_file = os.path.join(scan_name,'ID_'+slice+'.dcm')
                shutil.copyfile(source_file,target_file)
            dcm2nii(scan_name,os.path.join(target_filepath,scan_name+'.nii.gz'))
