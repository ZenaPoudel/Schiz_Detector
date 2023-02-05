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
    if mri_type == 'T1':
    mcic_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
    cobre_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
    mcic_participants = []
    mcic_path = []

    for path in mcic_file:
      mcic_participants.append(path.split('/')[7][4:])
      mcic_path.append(path)

    mcic_all_df = pd.DataFrame({
        'participant_id': mcic_participants, 
        'path': mcic_path
    })

    mcic_unique_df = mcic_all_df.drop_duplicates('participant_id', keep='first')

    mcic_unique = mcic_unique_df['path'].tolist()

    cobre_participants = []
    cobre_path = []

    for path in cobre_file:
      cobre_participants.append(path.split('/')[8][4:])
      cobre_path.append(path)

    cobre_all_df = pd.DataFrame({
        'participant_id': cobre_participants, 
        'path': cobre_path
    })

    cobre_unique_df = cobre_all_df.drop_duplicates('participant_id', keep='first')
    cobre_unique = cobre_unique_df['path'].tolist()


  #             file = np.concatenate((mcic_file, cobre_file))
  #             np.array(np.random.shuffle(file))
    elif mri_type == 'T2':
      mcic_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
      cobre_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)
      mcic_participants = []
      mcic_path = []

      for path in mcic_file:
        mcic_participants.append(path.split('/')[7][4:])
        mcic_path.append(path)

      mcic_all_df = pd.DataFrame({
          'participant_id': mcic_participants, 
          'path': mcic_path
      })
      mcic_unique_df = mcic_all_df.drop_duplicates('participant_id', keep='first')
      mcic_unique = mcic_unique_df['path'].tolist()

      cobre_participants = []
      cobre_path = []

      for path in cobre_file:
        cobre_participants.append(path.split('/')[8][4:])
        cobre_path.append(path)

      cobre_all_df = pd.DataFrame({
          'participant_id': cobre_participants, 
          'path': cobre_path
      })

      cobre_unique_df = cobre_all_df.drop_duplicates('participant_id', keep='first')
      cobre_unique = cobre_unique_df['path'].tolist()
      
  mcic_tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
  cobre_tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
  mcic_participants = pd.read_csv(mcic_tsv_path,sep='\t')
  cobre_participants = pd.read_csv(cobre_tsv_path,sep='\t')


  MCIC_healthy_subjects = []
  MCIC_schiz_subjects = []
  COBRE_healthy_subjects = []
  COBRE_schiz_subjects = []
  mcic_participants['diagnosis'] = np.where(mcic_participants['dx']== 'No_Known_Disorder', 0, 1)
  diagnosis=[]
  for index, row in cobre_participants.iterrows():
    if (row['dx'] == 'No_Known_Disorder'):
      diagnosis.append(0)
    elif (row['dx'] == 'Bipolar_Disorder'):
      diagnosis.append(2)
    else:
      diagnosis.append(1)

  cobre_participants = pd.concat([cobre_participants, pd.DataFrame(diagnosis)], axis=1)

  cobre_participants.columns = ['study', 'participant_id','age', 'sex','dx','diagnosis']

  for index, row in mcic_participants.iterrows():
    if (row['diagnosis'] == 1):
      MCIC_schiz_subjects.append(row['participant_id'])
    else:
      MCIC_healthy_subjects.append(row['participant_id'])
  for index, row in cobre_participants.iterrows():
    if (row['diagnosis'] == 1):
      COBRE_schiz_subjects.append(row['participant_id'])
    elif(row['diagnosis'] == 0):
      COBRE_healthy_subjects.append(row['participant_id'])

  MCIC_healthy = []
  MCIC_schiz = []
  MCIC_healthy_labels = []
  MCIC_schiz_labels = []
  COBRE_healthy = []
  COBRE_schiz = []
  COBRE_healthy_labels = []
  COBRE_schiz_labels = []

  for M in mcic_unique:
    if (any(ele in M for ele in MCIC_healthy_subjects)):
      MCIC_healthy.append(M)
      MCIC_healthy_labels.append(0)
    else:  
      MCIC_schiz.append(M)
      MCIC_schiz_labels.append(1)
  for C in cobre_unique:
    if (any(ele in C for ele in COBRE_healthy_subjects)):
      COBRE_healthy.append(C)
      COBRE_healthy_labels.append(0)
    elif(any(ele in cobre_t1 for ele in COBRE_schiz_subjects)):  
      COBRE_schiz.append(C)
      COBRE_schiz_labels.append(1)

  transforms = Compose([NormalizeIntensity, EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=pix_dimension), ResizeWithPadOrCrop(spatial_size=resize_spatial_size)])

  healthy_train = []
  healthy_valid = []
  schiz_train = []
  schiz_valid = []

  healthy_train_label = []
  healthy_valid_label = []
  schiz_train_label = []
  schiz_valid_label = []

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

  train_healthy_ds = ImageDataset(image_files= healthy_train, labels=healthy_train_label, image_only=True, transform=transforms)
  train_schiz_ds = ImageDataset(image_files=schiz_train , labels=schiz_train_label, image_only=True, transform=transforms)
  train_ds = train_healthy_ds + train_schiz_ds 

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,worker_init_fn=seed_worker, generator=g, pin_memory=pin_memory)


  # create a validation data loader
  val_healthy_ds = ImageDataset(image_files=healthy_val, labels=healthy_val_label, image_only=True, transform=transforms)
  val_schiz_ds = ImageDataset(image_files=schiz_val, labels=schiz_val_label, image_only=True,  transform=transforms)
  val_ds = val_healthy_ds + val_schiz_ds 

  val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)

  else:
    mcic_t1_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T1w.nii', recursive = True)
    cobre_t1_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T1w.nii.gz', recursive = True)
    
    mcic_t1_participants = []
    mcic_t1_path = []

    for path in mcic_t1_file:
      mcic_t1_participants.append(path.split('/')[7][4:])
      mcic_t1_path.append(path)

    mcic_t1_all_df = pd.DataFrame({
        'participant_id': mcic_t1_participants, 
        'path': mcic_t1_path
    })
    mcic_t1_unique_df = mcic_t1_all_df.drop_duplicates('participant_id', keep='first')
    mcic_t1_unique = mcic_t1_unique_df['path'].tolist()

    cobre_t1_participants = []
    cobre_t1_path = []

    for path in cobre_t1_file:
      cobre_t1_participants.append(path.split('/')[8][4:])
      cobre_t1_path.append(path)

    cobre_t1_all_df = pd.DataFrame({
        'participant_id': cobre_t1_participants, 
        'path': cobre_t1_path
    })

    cobre_t1_unique_df = cobre_t1_all_df.drop_duplicates('participant_id', keep='first')
    cobre_t1_unique = cobre_t1_unique_df['path'].tolist()
      
    mcic_t2_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/**/anat/*_T2w.nii', recursive = True)
    cobre_t2_file = glob2.glob('/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/**/anat/**/*_T2w.nii.gz', recursive = True)
    
    mcic_t2_participants = []
    mcic_t2_path = []

    for path in mcic_t2_file:
      mcic_t2_participants.append(path.split('/')[7][4:])
      mcic_t2_path.append(path)

    mcic_t2_all_df = pd.DataFrame({
        'participant_id': mcic_t2_participants, 
        'path': mcic_t2_path
    })

    mcic_t2_unique_df = mcic_t2_all_df.drop_duplicates('participant_id', keep='first')
    mcic_t2_unique = mcic_t2_unique_df['path'].tolist()
    
    
    cobre_t2_participants = []
    cobre_t2_path = []

    for path in cobre_t2_file:
      cobre_t2_participants.append(path.split('/')[8][4:])
      cobre_t2_path.append(path)

    cobre_t2_all_df = pd.DataFrame({
        'participant_id': cobre_t2_participants, 
        'path': cobre_t2_path
    })

    cobre_t2_unique_df = cobre_t2_all_df.drop_duplicates('participant_id', keep='first')
    cobre_t2_unique = cobre_t2_unique_df['path'].tolist()
    
    mcic_t1_t2 = []

    for t1 in mcic_t1_unique:
      for t2 in mcic_t2_unique:
        if t1.split('/')[7][4:] == t2.split('/')[7][4:]:
          mcic_t1_t2.append((t1, t2))
    
    cobre_t1_t2 = []

    for t1 in cobre_t1_unique:
      for t2 in cobre_t2_unique:
        if t1.split('/')[8][4:] == t2.split('/')[8][4:]:
          cobre_t1_t2.append((t1, t2))
    
    mcic_tsv_path = '/content/drive/MyDrive/schizophrenia_data/MCIC_/MCICShare/participants.tsv'
    cobre_tsv_path = '/content/drive/MyDrive/schizophrenia_data/COBRE/schizconnect_COBRE_images_16224/COBRE/participants.tsv'
    mcic_participants = pd.read_csv(mcic_tsv_path,sep='\t')
    cobre_participants = pd.read_csv(cobre_tsv_path,sep='\t')
    
    MCIC_healthy_subjects = []
    MCIC_schiz_subjects = []
    COBRE_healthy_subjects = []
    COBRE_schiz_subjects = []
    mcic_participants['diagnosis'] = np.where(mcic_participants['dx']== 'No_Known_Disorder', 0, 1)
    
    diagnosis=[]
    for index, row in cobre_participants.iterrows():
      if (row['dx'] == 'No_Known_Disorder'):
        diagnosis.append(0)
      elif (row['dx'] == 'Bipolar_Disorder'):
        diagnosis.append(2)
      else:
        diagnosis.append(1)

    cobre_participants = pd.concat([cobre_participants, pd.DataFrame(diagnosis)], axis=1)

    cobre_participants.columns = ['study', 'participant_id','age', 'sex','dx','diagnosis']


    for index, row in mcic_participants.iterrows():
      if (row['diagnosis'] == 1):
        MCIC_schiz_subjects.append(row['participant_id'])
      else:
        MCIC_healthy_subjects.append(row['participant_id'])
            
    for index, row in cobre_participants.iterrows():
      if (row['diagnosis'] == 1):
        COBRE_schiz_subjects.append(row['participant_id'])
      elif(row['diagnosis'] == 0):
        COBRE_healthy_subjects.append(row['participant_id'])
    
    
    MCIC_t1_t2_healthy = []
    MCIC_t1_t2_schiz = []
    MCIC_healthy_labels= []
    MCIC_schiz_labels= []

    COBRE_t1_t2_healthy = []
    COBRE_t1_t2_schiz = []
    COBRE_healthy_labels = []
    COBRE_schiz_labels = []      

    for M1,M2 in mcic_t1_t2:
      if (any(ele in M1 for ele in MCIC_healthy_subjects)):
        MCIC_t1_t2_healthy.append((M1,M2))
        MCIC_healthy_labels.append(0)
      else:  
        MCIC_t1_t2_schiz.append((M1,M2))
        MCIC_schiz_labels.append(1)
            
    for C1,C2 in cobre_t1_t2:
      if (any(ele in C1 for ele in COBRE_healthy_subjects)):
        COBRE_t1_t2_healthy.append((C1,C2))
        COBRE_healthy_labels.append(0)
      elif(any(ele in cobre_t1 for ele in COBRE_schiz_subjects)):  
        COBRE_t1_t2_schiz.append((C1,C2))
        COBRE_schiz_labels.append(1)
    
    if ratio=='Yes':
      transforms = Compose([LoadImage(image_only=True), ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=(2,2,2)), ResizeWithPadOrCrop(spatial_size=((99,99,99)))])
      MCIC_ratio_healthy = []
      MCIC_ratio_schiz = []
      COBRE_ratio_healthy = []
      COBRE_ratio_schiz = []
      for M1,M2 in MCIC_t1_t2_healthy:
        T1 = transforms(M1)
        T2 = transforms(M2)
        healthy_ratio = (T1-T2)/(T1+T2)
        healthy_ratio[healthy_ratio!=healthy_ratio] = 0
        healthy_ratio=ScaleIntensity()(healthy_ratio)
        MCIC_ratio_healthy.append(healthy_ratio)


      for M1,M2 in MCIC_t1_t2_schiz:
        T1 = transforms(M1)
        T2 = transforms(M2)
        schiz_ratio = (T1-T2)/(T1+T2)
        schiz_ratio[schiz_ratio!=schiz_ratio] = 0
        schiz_ratio=ScaleIntensity()(schiz_ratio)
        MCIC_ratio_schiz.append(schiz_ratio)
          
      for C1,C2 in COBRE_t1_t2_healthy:
        T1 = transforms(C1)
        T2 = transforms(C2)
        healthy_ratio = (T1-T2)/(T1+T2)
        healthy_ratio[healthy_ratio!=healthy_ratio] = 0
        healthy_ratio=ScaleIntensity()(healthy_ratio)
        COBRE_ratio_healthy.append(healthy_ratio)


      for M1,M2 in COBRE_t1_t2_schiz:
        T1 = transforms(C1)
        T2 = transforms(C2)
        schiz_ratio = (T1-T2)/(T1+T2)
        schiz_ratio[schiz_ratio!=schiz_ratio] = 0
        schiz_ratio=ScaleIntensity()(schiz_ratio)
        COBRE_ratio_schiz.append(schiz_ratio)
      
      healthy_train = []
      healthy_valid = []
      schiz_train = []
      schiz_valid = []

      healthy_train_label = []
      healthy_valid_label = []
      schiz_train_label = []
      schiz_valid_label = []
      
      MCIC_healthy_split = int(len(MCIC_ratio_healthy) * (1 - test_split))
      MCIC_schiz_split = int(len(MCIC_ratio_schiz) * (1 - test_split))
      
      COBRE_healthy_split = int(len(COBRE_ratio_healthy) * (1 - test_split))
      COBRE_schiz_split = int(len(COBRE_ratio_schiz) * (1 - test_split))
      
      healthy_train = np.concatenate((MCIC_ratio_healthy[:MCIC_healthy_split], COBRE_ratio_healthy[:COBRE_healthy_split]))
      schiz_train = np.concatenate((MCIC_ratio_schiz[:MCIC_schiz_split], COBRE_ratio_schiz[:COBRE_schiz_split]))
      healthy_val = np.concatenate((MCIC_ratio_healthy[MCIC_healthy_split:], COBRE_ratio_healthy[COBRE_healthy_split:]))
      schiz_val = np.concatenate((MCIC_ratio_schiz[MCIC_schiz_split:], COBRE_ratio_schiz[COBRE_schiz_split:]))

      healthy_train_label = np.concatenate((MCIC_healthy_labels[:MCIC_healthy_split], COBRE_healthy_labels[:COBRE_healthy_split]))
      schiz_train_label = np.concatenate((MCIC_schiz_labels[:MCIC_schiz_split], COBRE_schiz_labels[:COBRE_schiz_split]))
      healthy_val_label = np.concatenate((MCIC_healthy_labels[MCIC_healthy_split:], COBRE_healthy_labels[COBRE_healthy_split:]))
      schiz_val_label = np.concatenate((MCIC_schiz_labels[MCIC_schiz_split:], COBRE_schiz_labels[COBRE_schiz_split:]))

      def my_collate(batch):
        """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
        # item: a tuple of (img, label)
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data)
        target = torch.LongTensor(target)
        return [data, target]
      
      train_healthy_ds= ArrayDataset(healthy_train, labels=healthy_train_label)
      train_schiz_ds = ArrayDataset(schiz_train,labels=schiz_train_label)
    
      train_ds =train_healthy_ds+train_schiz_ds

      train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn = my_collate, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g, pin_memory=pin_memory)

      # create a validation data loader
      val_healthy_ds= ArrayDataset(healthy_val, labels=healthy_val_label)
      val_schiz_ds = ArrayDataset(schiz_val,labels=schiz_val_label)

      val_ds =val_healthy_ds+val_schiz_ds

      val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn = my_collate, num_workers=2, pin_memory=pin_memory)

    elif ratio=='channel':
      transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Orientation(axcodes='RAS'), Spacing(pixdim=(3,3,3)), ResizeWithPadOrCrop(spatial_size=(99,99,99))])            
      healthy_split = int(len(MCIC_t1_t2_healthy) * (1 - test_split))
      schiz_split = int(len(MCIC_t1_t2_schiz) * (1 - test_split))
      
      train_healthy_ds = TwoImageDataset(image_files=MCIC_t1_t2_healthy[:healthy_split], labels=MCIC_healthy_labels[:healthy_split], transform=transforms, image_only=True)
      train_schiz_ds = TwoImageDataset(image_files=MCIC_t1_t2_schiz[:schiz_split], labels=MCIC_schiz_labels[:schiz_split], transform=transforms, image_only=True)
      train_ds = train_healthy_ds + train_schiz_ds 
      
      train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g, pin_memory=pin_memory)
      # create a validation data loader
      val_healthy_ds = TwoImageDataset(image_files=MCIC_t1_t2_healthy[healthy_split:], labels=MCIC_healthy_labels[healthy_split:], transform=transforms, image_only=True)
      val_schiz_ds = TwoImageDataset(image_files=MCIC_t1_t2_schiz[schiz_split:], labels=MCIC_schiz_labels[schiz_split:], transform=transforms, image_only=True)
      val_ds = val_healthy_ds + val_schiz_ds 
      
      val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)
  return train_ds, train_loader, val_loader
