import numpy as np
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ResizeWithPadOrCrop,
    LoadImage,
    Spacing,
    ScaleIntensity,
    Orientation, 
)
import torch
import torch.nn as nn

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" ) #if torch.cuda.is_available() else "cpu")
def generate_cam (model, image_path, image_label, state_dict_path):
    def last_conv_layer():
        last_conv_layer_name = list(filter(lambda x: isinstance(x, nn.Conv3d), model.modules()))[-1]
        return last_conv_layer_name
    def forward_hook(module, input, output):
        activation.append(output)
    def backward_hook(module, grad_in, grad_out):
        grad.append(grad_out[0])
    
    last_conv_layer_name= last_conv_layer()
    last_conv_layer_name.register_forward_hook(forward_hook)
    last_conv_layer_name.register_backward_hook(backward_hook)
    grad = []
    activation = []
  
    transforms = Compose([LoadImage(ensure_channel_first=True, image_only=True),ScaleIntensity(), Orientation(axcodes='RAS'), Spacing(pixdim=(2,2,2)), ResizeWithPadOrCrop(spatial_size=(99,99,99))])

    # pred = ImageDataset(image_files=image_path, labels=image_label, image_only=True, transform=transforms)

    # pred = DataLoader(pred, pin_memory=pin_memory)
  
    image = transforms(image_path)

    label_pred = image_label

    # pred = monai.utils.misc.first(pred)
    # image , label_pred = pred[0].to(device), pred[1].to(device)
    
    print(model)
    print(state_dict_path)
    theModel = model.load_state_dict(torch.load(state_dict_path))
    print('oops')
    predicted = theModel(image).to(device)
    pred_labels = []
    for i in label_pred:
    # print(i.item())
        if i == 0:
            i = [1,0]
            pred_labels.append(i)
        elif i == 1:
            i = [0,1]
            pred_labels.append(i)
    pred_label = np.array(pred_labels)
    pred_labels = torch.from_numpy(pred_label)
    pred_labels = pred_labels.float().to(device)

    loss_function = torch.nn.BCELoss()

    pred_loss = loss_function(predicted, pred_labels)

    theModel.zero_grad()

    pred_loss.backward()

    grads = grad[0].cpu().data.numpy().squeeze()
    fmap = activation[0].cpu().data.numpy().squeeze()

    tmp = grads.reshape([grads.shape[0],-1])
    weights = np.mean(tmp, axis=1)

    cam = np.zeros(grads.shape[1:])
    for i,w in enumerate(weights):
        cam += w*fmap[i,:]
    cam = (cam>0)*cam
    cam = cam/cam.max()*255

    cam = EnsureChannelFirst()(cam)
    cam = Spacing(pixdim=(2,2,2))(cam)
    cam = Resize(spatial_size=(99,99,99))(cam)

    return image, cam
