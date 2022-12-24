# from google.colab import drive
# drive.mount("/content/drive")
# import os
# os.chdir("/content/drive/")
# !ls

# pip install nibabel -U

# pip install monai

import numpy as np
import pandas as pd
import glob2
# import matplotlib.pyplot as plt 
# import nibabel.processing as nib_processing
# import nibabel.affines as nib_affines
import matplotlib.pyplot as plt
import torch
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, ArrayDataset
from trainer.custom_dataset import TwoImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ResizeWithPadOrCrop,
    Spacing,
    ScaleIntensity,
    Orientation, 
    LoadImage,
    SpatialResample,
    Lambda,
    AffineGrid,
    Resample,
    EnsureType
)


pin_memory = torch.cuda.is_available() #MAIN


def data_pull_and_load(
    data_source= 'MCIC', 
    mri_type = 'T1', 
    pix_dimension=(2,2,2), 
    resize_spatial_size=(99,99,99),
    test_split=0.3,
    batch_size=8,
    ratio='NO'):
    if ratio == 'NO':
        if data_source == 'MCIC':
            if mri_type == 'T1':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
            elif mri_type == 'T2':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
            elif mri_type == 'both':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*.nii', recursive = True)
            tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
            participants = pd.read_csv(tsv_path,sep='\t')
        elif data_source == 'COBRE':
            if mri_type == 'T1':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
            elif mri_type == 'T2':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)
            elif mri_type == 'both':
                file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*.nii.gz', recursive = True)
            tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
            participants = pd.read_csv(tsv_path,sep='\t')
        elif data_source == 'both':
            if mri_type == 'T1':
                mcic_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
                cobre_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
        #             file = np.concatenate((mcic_file, cobre_file))
        #             np.array(np.random.shuffle(file))
            elif mri_type == 'T2':
                mcic_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
                cobre_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)
        #             file = np.concatenate((mcic_file, cobre_file))
        #             np.random.shuffle(file)
            elif mri_type == 'both':
                mcic_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*.nii', recursive = True)
                cobre_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*.nii.gz', recursive = True)
        #             file = np.concatenate((mcic_file, cobre_file))
        #             np.random.shuffle(file)
            mcic_tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
            cobre_tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
            mcic_participants = pd.read_csv(mcic_tsv_path,sep='\t')
            cobre_participants = pd.read_csv(cobre_tsv_path,sep='\t')

        if data_source == 'both':
            MCIC_healthy_subjects = []
            MCIC_schiz_subjects = []
            COBRE_healthy_subjects = []
            COBRE_schiz_subjects = []
            mcic_participants['diagnosis'] = np.where(mcic_participants['dx']== 'No_Known_Disorder', 0, 1)
            cobre_participants['diagnosis'] = np.where(cobre_participants['dx']== 'No_Known_Disorder', 0, 1)

            for index, row in mcic_participants.iterrows():
                if (row['diagnosis'] == 1):
                    MCIC_schiz_subjects.append(row['participant_id'])
                else:
                    MCIC_healthy_subjects.append(row['participant_id'])
            for index, row in cobre_participants.iterrows():
                if (row['diagnosis'] == 1):
                    COBRE_schiz_subjects.append(row['participant_id'])
                else:
                    COBRE_healthy_subjects.append(row['participant_id'])

            MCIC_healthy = []
            MCIC_schiz = []
            MCIC_healthy_labels = []
            MCIC_schiz_labels = []
            COBRE_healthy = []
            COBRE_schiz = []
            COBRE_healthy_labels = []
            COBRE_schiz_labels = []

            for M in mcic_file:
                if (any(ele in M for ele in MCIC_healthy_subjects)):
                    MCIC_healthy.append(M)
                    MCIC_healthy_labels.append(0)
                else:  
                    MCIC_schiz.append(M)
                    MCIC_schiz_labels.append(1)
            for C in cobre_file:
                if (any(ele in C for ele in COBRE_healthy_subjects)):
                    COBRE_healthy.append(C)
                    COBRE_healthy_labels.append(0)
                else:  
                    COBRE_schiz.append(C)
                    COBRE_schiz_labels.append(1)

            transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=pix_dimension), ResizeWithPadOrCrop(spatial_size=resize_spatial_size)])

            healthy_train = []
            healthy_valid = []
            schiz_train = []
            schiz_valid = []
            MCIC_healthy_split = int(len(MCIC_healthy) * (1 - test_split))
            MCIC_schiz_split = int(len(MCIC_schiz) * (1 - test_split))
            COBRE_healthy_split = int(len(COBRE_healthy) * (1 - test_split))
            COBRE_schiz_split = int(len(COBRE_schiz) * (1 - test_split))

            healthy_train = np.concatenate((MCIC_healthy[:MCIC_healthy_split], COBRE_healthy[:COBRE_healthy_split]))
            schiz_train = np.concatenate((MCIC_schiz[:MCIC_schiz_split], COBRE_schiz[:COBRE_schiz_split]))
            healthy_val = np.concatenate((MCIC_healthy[MCIC_healthy_split:], COBRE_healthy[COBRE_healthy_split:]))
            schiz_val = np.concatenate((MCIC_schiz[MCIC_schiz_split:], COBRE_schiz[COBRE_schiz_split:]))

            healthy_train_label = np.concatenate((MCIC_healthy_labels[:MCIC_healthy_split], COBRE_healthy_labels[:COBRE_healthy_split]))
            schiz_train_label = np.concatenate((MCIC_schiz_labels[:MCIC_schiz_split], COBRE_schiz_labels[:COBRE_schiz_split]))
            healthy_val_label = np.concatenate((MCIC_healthy_labels[MCIC_healthy_split:], COBRE_healthy_labels[COBRE_healthy_split:]))
            schiz_val_label = np.concatenate((MCIC_schiz_labels[MCIC_schiz_split:], COBRE_schiz_labels[COBRE_schiz_split:]))

        #         healthy_train = np.concatenate((MCIC_healthy[:30], COBRE_healthy[:30]))
        #         schiz_train = np.concatenate((MCIC_schiz[:30], COBRE_schiz[:30]))
        #         healthy_val = np.concatenate((MCIC_healthy[30:40], COBRE_healthy[30:40]))
        #         schiz_val = np.concatenate((MCIC_schiz[30:40], COBRE_schiz[30:40]))

        #         healthy_train_label = np.concatenate((MCIC_healthy_labels[:30], COBRE_healthy_labels[:30]))
        #         schiz_train_label = np.concatenate((MCIC_schiz_labels[:30], COBRE_schiz_labels[:30]))
        #         healthy_val_label = np.concatenate((MCIC_healthy_labels[30:40], COBRE_healthy_labels[30:40]))
        #         schiz_val_label = np.concatenate((MCIC_schiz[30:40], COBRE_schiz[30:40]))

            train_healthy_ds = ImageDataset(image_files= healthy_train, labels=healthy_train_label, image_only=True, transform=transforms)
            train_schiz_ds = ImageDataset(image_files=schiz_train , labels=schiz_train_label, image_only=True, transform=transforms)
            train_ds = train_healthy_ds + train_schiz_ds 

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
            # im2, label2 = monai.utils.misc.first(train_loader)
            # print(type(im2), im2.shape, label2, label2.shape)

            # create a validation data loader
            val_healthy_ds = ImageDataset(image_files=healthy_val, labels=healthy_val_label, image_only=True, transform=transforms)
            val_schiz_ds = ImageDataset(image_files=schiz_val, labels=schiz_val_label, image_only=True,  transform=transforms)
            val_ds = val_healthy_ds + val_schiz_ds 

            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
        else:
            healthy_subjects = []
            schiz_subjects = []

            participants['diagnosis'] = np.where(participants['dx']== 'No_Known_Disorder', 0, 1)

            for index, row in participants.iterrows():
                if (row['diagnosis'] == 1):
                    schiz_subjects.append(row['participant_id'])
                else:
                    healthy_subjects.append(row['participant_id'])

            healthy = []
            schiz = []
            healthy_labels = []
            schiz_labels = []

            for f in file:
                if (any(ele in f for ele in healthy_subjects)):
                    healthy.append(f)
                    healthy_labels.append(0)
                else:  
                    schiz.append(f)
                    schiz_labels.append(1)

            transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=pix_dimension), ResizeWithPadOrCrop(spatial_size=resize_spatial_size)])
        #     print("Resize instead of ResizeWithPadOrCrop")
        #     transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=pix_dimension), Resize(spatial_size=resize_spatial_size)])

            healthy_split = int(len(healthy) * (1 - test_split))
            schiz_split = int(len(schiz) * (1 - test_split))

            train_healthy_ds = ImageDataset(image_files=healthy[:healthy_split], labels=healthy_labels[:healthy_split], image_only=True, transform=transforms)
            train_schiz_ds = ImageDataset(image_files=schiz[:schiz_split], labels=schiz_labels[:schiz_split],image_only=True, transform=transforms)
            train_ds = train_healthy_ds + train_schiz_ds 
            # type(train_ds) 

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
            # im2, label2 = monai.utils.misc.first(train_loader)
            # print(type(im2), im2.shape, label2, label2.shape)

            # create a validation data loader
            val_healthy_ds = ImageDataset(image_files=healthy[healthy_split:], labels=healthy_labels[healthy_split:], image_only=True, transform=transforms)
            val_schiz_ds = ImageDataset(image_files=schiz[schiz_split:], labels=schiz_labels[schiz_split:], image_only=True,  transform=transforms)
            val_ds = val_healthy_ds + val_schiz_ds 

            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    else:
        mcic_t1_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
        cobre_t1_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)

        mcic_t2_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
        cobre_t2_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)

        mcic_tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
        cobre_tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
        mcic_participants = pd.read_csv(mcic_tsv_path,sep='\t')
        cobre_participants = pd.read_csv(cobre_tsv_path,sep='\t')
        same_subject_and_run_t1_t2_path= []
        
        MCIC_healthy_subjects = []
        MCIC_schiz_subjects = []
#         COBRE_healthy_subjects = []
#         COBRE_schiz_subjects = []
        mcic_participants['diagnosis'] = np.where(mcic_participants['dx']== 'No_Known_Disorder', 0, 1)
        # cobre_participants['diagnosis'] = np.where(cobre_participants['dx']== 'No_Known_Disorder', 0, 1)

        for index, row in mcic_participants.iterrows():
            if (row['diagnosis'] == 1):
                MCIC_schiz_subjects.append(row['participant_id'])
            else:
                MCIC_healthy_subjects.append(row['participant_id'])
                
        same_subject_t1_t2_path= []

#         for i in mcic_t1_file:
#             t1_subject = i.split('/')[7]

#             for j in mcic_t2_file:
#                 t2_subject = j.split('/')[7]

#                 if (t1_subject == t2_subject):
#                     similar_path = (i, j)
#                     same_subject_t1_t2_path.append(similar_path)
        for i in mcic_t1_file:
            t1_subject = i.split('/')[7]
            t1_run = i.split('/')[-1].split('_')[-2]

            for j in mcic_t2_file:
                t2_subject = j.split('/')[7]
                t2_run = j.split('/')[-1].split('_')[-2]

                if ( (t1_subject == t2_subject) and (t1_run == t2_run)): 
                    similar_path = (i, j)
                    same_subject_and_run_t1_t2_path.append(similar_path)
        MCIC_t1_t2_healthy = []
        MCIC_t1_t2_schiz = []
        MCIC_healthy_labels= []
        MCIC_schiz_labels= []

#             COBRE_t1_t2_healthy = []
#             COBRE_t1_t2_schiz = []
#             COBRE_healthy_labels = []
#             COBRE_schiz_labels = []      
     

        if ratio=='Yes':
           for M1,M2 in same_subject_and_run_t1_t2_path:
            if (any(ele in M1 for ele in MCIC_healthy_subjects)):
                MCIC_t1_t2_healthy.append((M1,M2))
            else:  
                MCIC_t1_t2_schiz.append((M1,M2))
            transforms = Compose([LoadImage(image_only=True), ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=(2,2,2)), ResizeWithPadOrCrop(spatial_size=((99,99,99)))])
            MCIC_ratio_healthy = []
            MCIC_ratio_schiz = []
            for M1,M2 in MCIC_t1_t2_healthy:
                T1 = transforms(M1)
                T2 = transforms(M2)
                healthy_ratio = (T1-T2)/(T1+T2)
                healthy_ratio[healthy_ratio!=healthy_ratio] = 0
                healthy_ratio=ScaleIntensity()(healthy_ratio)
                MCIC_ratio_healthy.append(healthy_ratio)
                MCIC_healthy_labels.append(0)


            for M1,M2 in MCIC_t1_t2_schiz:
                T1 = transforms(M1)
                T2 = transforms(M2)
                schiz_ratio = (T1-T2)/(T1+T2)
                schiz_ratio[schiz_ratio!=schiz_ratio] = 0
                schiz_ratio=ScaleIntensity()(schiz_ratio)
                MCIC_ratio_schiz.append(schiz_ratio)
                MCIC_schiz_labels.append(1)
            
            healthy_split = int(len(MCIC_ratio_healthy) * (1 - test_split))
            schiz_split = int(len(MCIC_ratio_schiz) * (1 - test_split))
            
            def my_collate(batch):
                """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
                # item: a tuple of (img, label)
                data = [item[0] for item in batch]
                target = [item[1] for item in batch]
                data = torch.stack(data)
                target = torch.LongTensor(target)
                return [data, target]
            
            train_healthy_ds= ArrayDataset(MCIC_ratio_healthy[:healthy_split], labels=MCIC_healthy_labels[:healthy_split])
            train_schiz_ds = ArrayDataset(MCIC_ratio_schiz[:schiz_split],labels=MCIC_schiz_labels[:schiz_split])

            # train_healthy_ds = ((MCIC_ratio_healthy[:healthy_split], MCIC_healthy_labels[:healthy_split]))
            # train_schiz_ds = ((MCIC_ratio_schiz[:schiz_split],MCIC_schiz_labels[:schiz_split]))
            # print(train_healthy_ds[1][10])
            train_ds =train_healthy_ds+train_schiz_ds

            train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn = my_collate, shuffle=True, num_workers=2, pin_memory=pin_memory)

            # create a validation data loader
            val_healthy_ds= ArrayDataset(MCIC_ratio_healthy[healthy_split:], labels=MCIC_healthy_labels[healthy_split:])
            val_schiz_ds = ArrayDataset(MCIC_ratio_schiz[schiz_split:],labels=MCIC_schiz_labels[schiz_split:])

            val_ds =val_healthy_ds+val_schiz_ds

            val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn = my_collate, shuffle=True, num_workers=2, pin_memory=pin_memory)

        elif ratio=='channel':
            for M1,M2 in same_subject_and_run_t1_t2_path:
                if (any(ele in M1 for ele in MCIC_healthy_subjects)):
                    MCIC_t1_t2_healthy.append((M1,M2))
                    MCIC_healthy_labels.append(0)
                else:  
                    MCIC_t1_t2_schiz.append((M1,M2))
                    MCIC_schiz_labels.append(1)
            transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=(3,3,3)), ResizeWithPadOrCrop(spatial_size=(99,99,99))])            
            healthy_split = int(len(MCIC_channel_healthy) * (1 - test_split))
            schiz_split = int(len(MCIC_channel_schiz) * (1 - test_split))
            
            train_healthy_ds = TwoImageDataset(image_files=MCIC_t1_t2_healthy[:healthy_split], labels=MCIC_healthy_labels[:healthy_split], transform=transforms, image_only=True)
            train_schiz_ds = TwoImageDataset(image_files=MCIC_t1_t2_schiz[:schiz_split], labels=MCIC_schiz_labels[:schiz_split], transform=transforms, image_only=True)
            train_ds = train_healthy_ds + train_schiz_ds 
            
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin_memory)
            # create a validation data loader
            val_healthy_ds = TwoImageDataset(image_files=MCIC_t1_t2_healthy[healthy_split:], labels=MCIC_healthy_labels[healthy_split:], transform=transforms, image_only=True)
            val_schiz_ds = TwoImageDataset(image_files=MCIC_t1_t2_schiz[schiz_split:], labels=MCIC_schiz_labels[schiz_split:], transform=transforms, image_only=True)
            val_ds = val_healthy_ds + val_schiz_ds 
            
            val_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin_memory)
    return train_ds, train_loader, val_loader
