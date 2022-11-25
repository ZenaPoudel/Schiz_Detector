import argparse
from grad_cam import generate_cam 

import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy.ma.core import masked_inside
sys.path.insert(0,'/content/Schiz_Detector/trainer')
from model import model_3DCNN
def main():
    parser = argparse.ArgumentParser(description='visualise the heatmap as per the slice number provided')
    parser.add_argument('--slice_num', type=int, default=45)
    parser.add_argument('--visualize', type=str, default='2d')
    parser.add_argument('--view', type = str, default='axial')
    parser.add_argument('--image_path', type=str, default='./')
    parser.add_argument('--image_label', type=int, default=1)
    parser.add_argument('--state_dict_path', type=str, default='./')
    parser.add_argument('--dropout', type=int, default=0.3)

    args = parser.parse_args()
    model = model_3DCNN(args.dropout)
    
    image, cam = generate_cam(model, args.image_path, args.image_label, args.state_dict_path)

    heat_map = image.cpu()*0.7+ cam*0.3

    if args.visualize== '2d':
        test_heat = heat_map[0,0,:,:,args.slice_num].cpu()
        plt.imshow(test_heat)
        plt.show()
    elif args.visualize== '3d':
        a,b,x,y,z = heat_map.shape
        frames = []
        
        if (args.view == 'axial'):
            for i in range(0, x):
                frame = heat_map[0,0,:,:,i]
                frames.append(frame) 
        elif (args.view == 'coronal'):
            for i in range(0, y):
                frame = heat_map[0,0,:,i,:]
                frames.append(frame)
        elif (args.view == 'sagittal'):
            for i in range(0, z):
                frame = heat_map[0,0,i,:,:]
                frames.append(frame)

        frames = np.array(frames)
        for i in frames:
            plt.imshow(i,plt.cm.gray)
            plt.show(block=False)
            plt.pause(0.3)
            plt.close()  
if __name__ == '__main__':
	main()
