import numpy as np
import pandas as pd
import random
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
from custom_dataset import TwoImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ResizeWithPadOrCrop,
    Spacing,
    ScaleIntensity,
    NormalizeIntensity,
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
    pix_dimension=(2,2,2), 
    resize_spatial_size=(99,99,99),
    test_split=0.3,
    batch_size=8,
    ratio='NO'):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    if ratio == 'NO':
        mcic_T1 = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
        mcic_tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
        mcic_participants = pd.read_csv(mcic_tsv_path,sep='\t')
        mcic_participants['diagnosis'] = np.where(mcic_participants['dx']== 'No_Known_Disorder', 0, 1)

        MCIC_healthy_subjects = []
        MCIC_schiz_subjects = []

        for index, row in mcic_participants.iterrows():
            if (row['diagnosis'] == 1):
                MCIC_schiz_subjects.append(row['participant_id'])
            else:
                MCIC_healthy_subjects.append(row['participant_id'])
        MCIC_healthy = []
        MCIC_schiz = []
        MCIC_healthy_labels = []
        MCIC_schiz_labels = []

        for M in mcic_T1:
            if (any(ele in M for ele in MCIC_healthy_subjects)):
                MCIC_healthy.append(M)
                MCIC_healthy_labels.append(0)
            else:  
                MCIC_schiz.append(M)
                MCIC_schiz_labels.append(1)
        
        
        print('MCIC Healthy Control:')
        print(len(MCIC_healthy))
        print('MCIC Schz Patients:')
        print(len(MCIC_schiz))

        MCIC_healthy_split = int(len(MCIC_healthy) * (1 - test_split))
        MCIC_schiz_split = int(len(MCIC_schiz) * (1 - test_split))

        print('MCIC Healthy train split')
        print(MCIC_healthy_split)
        print('MCICHealthy Val split')
        print(len(MCIC_healthy)-MCIC_healthy_split)
        print('MCIC Schiz train split')
        print(MCIC_schiz_split)
        print('MCIC Schiz VAL split')
        print(len(MCIC_schiz)-MCIC_schiz_split)

        cobre_T1 = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
        cobre_tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
        cobre_participants = pd.read_csv(cobre_tsv_path,sep='\t')

        diagnosis = []
        for index, row in cobre_participants.iterrows():
            if (row['dx'] == 'No_Known_Disorder'):
                diagnosis.append(0)
            elif (row['dx'] == 'Bipolar_Disorder'):
                diagnosis.append(2)
            else:
                diagnosis.append(1)

        cobre_participants = pd.concat([cobre_participants, pd.DataFrame(diagnosis)], axis=1)
        cobre_participants.columns = ['study', 'participant_id','age', 'sex','dx','diagnosis']

        COBRE_healthy_subjects = []
        COBRE_schiz_subjects = []

        for index, row in cobre_participants.iterrows():
            if (row['diagnosis'] == 1):
                COBRE_schiz_subjects.append(row['participant_id'])
            elif (row['diagnosis'] == 0):
                COBRE_healthy_subjects.append(row['participant_id'])

        COBRE_healthy = []
        COBRE_schiz = []
        COBRE_healthy_labels = []
        COBRE_schiz_labels = []

        for C in cobre_T1:
            if (any(ele in C for ele in COBRE_healthy_subjects)):
                COBRE_healthy.append(C)
                COBRE_healthy_labels.append(0)
            elif (any(ele in C for ele in COBRE_schiz_subjects)):
                COBRE_schiz.append(C)
                COBRE_schiz_labels.append(1)

        
        print('COBRE Healthy Control:')
        print(len(COBRE_healthy))
        print('COBRE Schz Patients:')
        print(len(COBRE_schiz))

        COBRE_healthy_split = int(len(COBRE_healthy) * (1 - test_split))
        COBRE_schiz_split = int(len(COBRE_schiz) * (1 - test_split))
        
        print('COBRE Healthy train split')
        print(COBRE_healthy_split)
        print('COBRE Healthy Val split')
        print(len(COBRE_healthy)-COBRE_healthy_split)
        print('COBRE SCHZ train split')
        print(COBRE_schiz_split)
        print('COBRE Schiz VAL split')
        print(len(COBRE_schiz)-COBRE_schiz_split)

        bgsc= glob2.glob('/content/drive/MyDrive/schizophrenia_data/unrepeated_bgsc_t1/**/*.nii.gz', recursive = True)
        bgsc_tsv_path = '/content/drive/MyDrive/schizophrenia_data/unrepeated_bgsc_t1/assessment_data/2079_Demographics_20220620.csv'
        bgsc_participants = pd.read_csv(bgsc_tsv_path,sep=',')
        diagnosis=[]

        for index, row in bgsc_participants.iterrows():
            if (row['Subject Type'] == 'Old Control'):
                diagnosis.append(0)
            elif (row['Subject Type'] == 'Young Control'):
                diagnosis.append(0)
            elif (row['Subject Type'] == 'Control'):
                diagnosis.append(0)
            else:
                diagnosis.append(1)
        bgsc_participants = pd.concat([bgsc_participants, pd.DataFrame(diagnosis)], axis=1)

        bgsc_healthy_subjects = []
        bgsc_schiz_subjects = []

        for index, row in bgsc_participants.iterrows():
            if (row[0] == 1):
                bgsc_schiz_subjects.append(row['Anonymized ID'])
            elif(row[0] == 0):
                bgsc_healthy_subjects.append(row['Anonymized ID'])

        bgsc_healthy = []
        bgsc_schiz = []
        bgsc_healthy_labels = []
        bgsc_schiz_labels = []

        for b in bgsc:
            if (any(ele in b for ele in bgsc_healthy_subjects)):
                bgsc_healthy.append(b)
                bgsc_healthy_labels.append(0)
            elif(any(ele in b for ele in bgsc_schiz_subjects)):    
                bgsc_schiz.append(b)
                bgsc_schiz_labels.append(1)

        print('BGSC Healthy Control:')
        print(len(bgsc_healthy))
        print('BGSC Schz Patients:')
        print(len(bgsc_schiz))
                        
        bgsc_healthy_split = int(len(bgsc_healthy) * (1 - test_split))
        bgsc_schiz_split = int(len(bgsc_schiz) * (1 - test_split))

        print('BGSC Healthy train split')
        print(bgsc_healthy_split)
        print('BGSC Healthy Val split')
        print(len(bgsc_healthy)-bgsc_healthy_split)
        print('BGSC SCHZ train split')
        print(bgsc_schiz_split)
        print('COBRE Schiz VAL split')
        print(len(bgsc_schiz)-bgsc_schiz_split)


        srpbs_healthy = glob2.glob('/content/drive/MyDrive/schizophrenia_data/SRPBS_/srpbs_healthy/**/t1/*.nii', recursive = True)
        srpbs_schiz = glob2.glob('/content/drive/MyDrive/schizophrenia_data/SRPBS_/srpbs_schiz/**/t1/*.nii', recursive = True)

        srpbs_healthy_labels = []
        srpbs_schiz_labels = []

        for s in srpbs_healthy:
            srpbs_healthy_labels.append(0)

        for s in srpbs_schiz:
            srpbs_schiz_labels.append(1)


        print('SRPBS Healthy Control:')
        print(len(srpbs_healthy))
        print('SRPBS SCHZ Patients:')
        print(len(srpbs_schiz))

        srpbs_healthy_split = int(len(srpbs_healthy) * (1 - test_split))
        srpbs_schiz_split = int(len(srpbs_schiz) * (1 - test_split))

        print('SRPBS Healthy train split')
        print(srpbs_healthy_split)
        print('SRPBS Healthy Val split')
        print(len(srpbs_healthy)-srpbs_healthy_split)
        print('as we have highly imbalanced data, we will be using the same number of healthy scontrol as the schz patients give below:')
        print('SRPBS SCHZ train split')
        print(srpbs_schiz_split)
        print('SRPBS Schiz VAL split')
        print(len(srpbs_schiz)-srpbs_schiz_split)


        transforms = Compose([NormalizeIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=(2,2,2)), ResizeWithPadOrCrop(spatial_size=(99,99,99))])

        healthy_train = np.concatenate((MCIC_healthy[:MCIC_healthy_split], COBRE_healthy[:COBRE_healthy_split], bgsc_healthy[:bgsc_healthy_split], srpbs_healthy[:srpbs_schiz_split]))
        schiz_train = np.concatenate((MCIC_schiz[:MCIC_schiz_split], COBRE_schiz[:COBRE_schiz_split], bgsc_schiz[:bgsc_schiz_split], srpbs_schiz[:srpbs_schiz_split]))

        healthy_val = np.concatenate((MCIC_healthy[MCIC_healthy_split:], COBRE_healthy[COBRE_healthy_split:], bgsc_healthy[bgsc_healthy_split:], srpbs_healthy[srpbs_schiz_split:len(srpbs_schiz)]))
        schiz_val = np.concatenate((MCIC_schiz[MCIC_schiz_split:], COBRE_schiz[COBRE_schiz_split:], bgsc_schiz[bgsc_schiz_split:], srpbs_schiz[srpbs_schiz_split:]))
        
        healthy_train_labels = np.concatenate((MCIC_healthy_labels[:MCIC_healthy_split], COBRE_healthy_labels[:COBRE_healthy_split], bgsc_healthy_labels[:bgsc_healthy_split], srpbs_healthy_labels[:srpbs_healthy_split]))
        schiz_train_labels = np.concatenate((MCIC_schiz_labels[:MCIC_schiz_split], COBRE_schiz_labels[:COBRE_schiz_split], bgsc_schiz_labels[:bgsc_schiz_split], srpbs_schiz_labels[:srpbs_schiz_split]))

        healthy_val_labels = np.concatenate((MCIC_healthy_labels[MCIC_healthy_split:], COBRE_healthy_labels[COBRE_healthy_split:], bgsc_healthy_labels[bgsc_healthy_split:], srpbs_healthy_labels[srpbs_healthy_split:]))
        schiz_val_labels = np.concatenate((MCIC_schiz_labels[MCIC_schiz_split:], COBRE_schiz_labels[COBRE_schiz_split:], bgsc_schiz_labels[bgsc_schiz_split:], srpbs_schiz_labels[srpbs_schiz_split:]))


        train_healthy_ds = ImageDataset(image_files=healthy_train, labels=healthy_train_labels, image_only=True, transform=transforms)
        train_schiz_ds = ImageDataset(image_files=schiz_train, labels=schiz_train_labels, image_only=True, transform=transforms)
        train_ds = train_healthy_ds + train_schiz_ds 

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,worker_init_fn=seed_worker, generator=g, pin_memory=pin_memory)

        val_healthy_ds = ImageDataset(image_files=healthy_val, labels=healthy_val_labels, image_only=True, transform=transforms)
        val_schiz_ds = ImageDataset(image_files=schiz_val, labels=schiz_val_labels, image_only=True,  transform=transforms)
        val_ds = val_healthy_ds + val_schiz_ds

        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)
            
    return train_ds, train_loader, val_loader
