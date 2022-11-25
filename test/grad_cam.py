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
  
    image = LoadImage(ensure_channel_first=True, image_only=True)(image_path)
    image = ScaleIntensity()(image)
    image = Orientation(axcodes='RAS')(image)
    image = Spacing(pixdim=(2,2,2))(image)
    image = ResizeWithPadOrCrop(spatial_size=(99,99,99))(image)
    image = image.unsqueeze_(0)
    image = image.cuda()

    label_pred = image_label
    model = model.cuda()
    model.load_state_dict(torch.load(state_dict_path))
    
    model.eval()
    predicted = model(image)

    pred_labels = []
    if label_pred == 0:
        label_pred = [1,0]
        pred_labels.append(label_pred)
    elif label_pred == 1:
        label_pred = [0,1]
        pred_labels.append(label_pred)
    pred_label = np.array(pred_labels)
    pred_labels = torch.from_numpy(pred_label)
    pred_labels = pred_labels.float().to(device)

    loss_function = torch.nn.BCELoss()

    pred_loss = loss_function(predicted, pred_labels)

    model.zero_grad()

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
