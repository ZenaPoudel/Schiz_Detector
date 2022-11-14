# from google.colab import drive
# drive.mount("/content/gdrive")
# import os
# os.chdir("/content/gdrive/")
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
from monai.data import DataLoader, ImageDataset
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
    Resample
)


pin_memory = torch.cuda.is_available() #MAIN


def data_pull_and_load(
    data_source= 'MCIC', 
    mri_type = 'T1', 
    pix_dimension=(2,2,2), 
    resize_spatial_size=(99,99,99),
    test_split=0.3,
    batch_size=8):
    
    if data_source == 'MCIC':
        if mri_type == 'T1':
            file = glob2.glob('/content/gdrive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
        elif mri_type == 'T2':
            file = glob2.glob('/content/gdrive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
        tsv_path = '/content/gdrive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
    
    elif data_source == 'COBRE':
        if mri_type == 'T1':
            file = glob.glob('/content/gdrive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
        elif mri_type == 'T2':
            file = glob.glob('/content/gdrive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)
        tsv_path = '/content/gdrive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
    

    participants = pd.read_csv(tsv_path,sep='\t')

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

    healthy_split = len(healthy) // (1.0 - test_split)
    schiz_split = len(schiz) // (1.0 - test_split)

    train_healthy_ds = ImageDataset(image_files=healthy[:healthy_split], labels=healthy_labels[:healthy_split], image_only=True, transform=transforms)
    train_schiz_ds = ImageDataset(image_files=schiz[:schiz_split], labels=schiz_labels[:schiz_split],image_only=True, transform=transforms)
    train_ds = train_healthy_ds + train_schiz_ds 
    # type(train_ds) 

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin_memory)
    # im2, label2 = monai.utils.misc.first(train_loader)
    # print(type(im2), im2.shape, label2, label2.shape)

    # create a validation data loader
    val_healthy_ds = ImageDataset(image_files=healthy[healthy_split:], labels=healthy_labels[healthy_split:], image_only=True, transform=transforms)
    val_schiz_ds = ImageDataset(image_files=schiz[schiz_split:], labels=schiz_labels[schiz_split:], image_only=True,  transform=transforms)
    val_ds = val_healthy_ds + val_schiz_ds 

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    
    return train_ds, train_loader, val_loader