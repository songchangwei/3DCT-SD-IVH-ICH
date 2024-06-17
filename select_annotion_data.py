import jsonlines,json
import os,shutil

with jsonlines.open('annotion_2.jsonl','r') as reader:
    for item in jsonlines.Reader(reader):
        print(item['patient_name'],item['patient_label'])
        for scan in item['scan_list']:
            print(scan['scan_name'],scan['scan_lable'])
            if scan['scan_lable'][0] == 1:
                base_dir = 'F:\\barin_chuxue_dcm\label_0'
                root_dir = 'I:\脑出血\\rsna-intracranial-hemorrhage-detection\\rsna-intracranial-hemorrhage-detection\stage_2_train'
                if not os.path.exists(base_dir+'/'+item['patient_name']):
                    os.mkdir(base_dir+'/'+item['patient_name'])
                if not os.path.exists(base_dir+'/'+item['patient_name']+'/'+scan['scan_name']):
                    os.mkdir(base_dir+'/'+item['patient_name']+'/'+scan['scan_name'])
                for slice in scan['slice_list']:
                    print(slice['slice_name'],slice['slice_label'],slice['slice_no'])
                    source_file = root_dir+'/ID_'+slice['slice_name']+'.dcm'
                    target_file = base_dir+'/'+item['patient_name']+'/'+scan['scan_name']+'/'+slice['slice_name']+'.dcm'
                    shutil.copyfile(source_file,target_file)
            
            if scan['scan_lable'][1] == 1:
                base_dir = 'F:\\barin_chuxue_dcm\label_1'
                root_dir = 'I:\脑出血\\rsna-intracranial-hemorrhage-detection\\rsna-intracranial-hemorrhage-detection\stage_2_train'
                if not os.path.exists(base_dir+'/'+item['patient_name']):
                    os.mkdir(base_dir+'/'+item['patient_name'])
                if not os.path.exists(base_dir+'/'+item['patient_name']+'/'+scan['scan_name']):
                    os.mkdir(base_dir+'/'+item['patient_name']+'/'+scan['scan_name'])
                for slice in scan['slice_list']:
                    print(slice['slice_name'],slice['slice_label'],slice['slice_no'])
                    source_file = root_dir+'/ID_'+slice['slice_name']+'.dcm'
                    target_file = base_dir+'/'+item['patient_name']+'/'+scan['scan_name']+'/'+slice['slice_name']+'.dcm'
                    shutil.copyfile(source_file,target_file)

            if scan['scan_lable'][2] == 1:
                base_dir = 'F:\\barin_chuxue_dcm\label_2'
                root_dir = 'I:\脑出血\\rsna-intracranial-hemorrhage-detection\\rsna-intracranial-hemorrhage-detection\stage_2_train'
                if not os.path.exists(base_dir+'/'+item['patient_name']):
                    os.mkdir(base_dir+'/'+item['patient_name'])
                if not os.path.exists(base_dir+'/'+item['patient_name']+'/'+scan['scan_name']):
                    os.mkdir(base_dir+'/'+item['patient_name']+'/'+scan['scan_name'])
                for slice in scan['slice_list']:
                    print(slice['slice_name'],slice['slice_label'],slice['slice_no'])
                    source_file = root_dir+'/ID_'+slice['slice_name']+'.dcm'
                    target_file = base_dir+'/'+item['patient_name']+'/'+scan['scan_name']+'/'+slice['slice_name'] +'.dcm'
                    shutil.copyfile(source_file,target_file)
            
            if scan['scan_lable'][3] == 1:
                base_dir = 'F:\\barin_chuxue_dcm\label_3'
                root_dir = 'I:\脑出血\\rsna-intracranial-hemorrhage-detection\\rsna-intracranial-hemorrhage-detection\stage_2_train'
                if not os.path.exists(base_dir+'/'+item['patient_name']):
                    os.mkdir(base_dir+'/'+item['patient_name'])
                if not os.path.exists(base_dir+'/'+item['patient_name']+'/'+scan['scan_name']):
                    os.mkdir(base_dir+'/'+item['patient_name']+'/'+scan['scan_name'])
                for slice in scan['slice_list']:
                    print(slice['slice_name'],slice['slice_label'],slice['slice_no'])
                    source_file = root_dir+'/ID_'+slice['slice_name']+'.dcm'
                    target_file = base_dir+'/'+item['patient_name']+'/'+scan['scan_name']+'/'+slice['slice_name'] +'.dcm'
                    shutil.copyfile(source_file,target_file)
             
            if scan['scan_lable'][4] == 1:
                base_dir = 'F:\\barin_chuxue_dcm\label_4'
                root_dir = 'I:\脑出血\\rsna-intracranial-hemorrhage-detection\\rsna-intracranial-hemorrhage-detection\stage_2_train'
                if not os.path.exists(base_dir+'/'+item['patient_name']):
                    os.mkdir(base_dir+'/'+item['patient_name'])
                if not os.path.exists(base_dir+'/'+item['patient_name']+'/'+scan['scan_name']):
                    os.mkdir(base_dir+'/'+item['patient_name']+'/'+scan['scan_name'])
                for slice in scan['slice_list']:
                    print(slice['slice_name'],slice['slice_label'],slice['slice_no'])
                    source_file = root_dir+'/ID_'+slice['slice_name']+'.dcm'
                    target_file = base_dir+'/'+item['patient_name']+'/'+scan['scan_name']+'/'+slice['slice_name'] +'.dcm'
                    shutil.copyfile(source_file,target_file)
                



        

