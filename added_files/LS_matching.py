import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
from util import util
import numpy as np
import cv2
from matplotlib import pyplot as plt

# To run: python3 /home/mcube/pytorch-CycleGAN-and-pix2pix/LS_publiser.py --dataroot /home/mcube/data_brick/data_brick_daolin/ --name tactile_nov_27_only_L1  --model pix2pix --direction AtoB --ntest 100 --num_test 1000


#############################################################################################################
#######################THINGS ADDED ERIC ####################################################################

object_name = "curved"
batch_size = 256
model_name = 'curved.pt'
pixels = 235

import torch
import sys
sys.path.insert(1, '/home/mcube/tactile_localization/ML/')
sys.path.insert(1, '/home/mcube/tactile_localization/src/')
import model
from camera_params import * # Loads virtual camera

from torch.utils.data import Dataset, TensorDataset

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split



from glob import glob

encodings = []

loc_dir = '/home/mcube/tactile_localization/'

import sys
sys.path.append(loc_dir + '/src/')
#import helper
from camera_params import fx, fy



def get_points_from_depth (depth):
    xv, yv = np.meshgrid(np.arange(depth.shape[0])-(depth.shape[0]-1)/2.0, np.arange(depth.shape[1])-(depth.shape[1]-1)/2.0, sparse=False, indexing='ij')
    xv = np.expand_dims(xv.flatten(), -1)
    yv = np.expand_dims(yv.flatten(),-1)
    points = np.expand_dims(depth.flatten(),-1)
    basic_points = np.concatenate([xv, yv], axis = -1)
    auxpoints = np.concatenate([basic_points, points], axis = -1)
    return auxpoints
    
def get_world_pointcloud(depth,fx,fy):
    pointss  = get_points_from_depth(depth)
    max_val = np.amax(depth); min_val = 0
    non_empty = pointss[:,2] < max_val-min_val
    empty = pointss[:,2] >= max_val-min_val

    pointss[:,0] *= pointss[:,2]
    pointss[:,1] *= pointss[:,2]
    pointss[:,0] /= fx
    pointss[:,1] /= fy
    #pointss[:,0] += (cx-cy)/fx

    aux = np.copy(pointss[:,1])
    pointss[:,1] = pointss[:,0]
    pointss[:,0] = aux
    reduced_points = pointss[non_empty]
    flat_points = pointss[non_empty]; flat_points[:,2] = 0.02
    pointss[empty,2] = 0
    flat_all_points = np.copy(pointss); flat_all_points[non_empty,2] = 0.02
    only_neg = pointss[empty]
    return reduced_points, pointss, flat_points, only_neg, flat_all_points



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.413, 0.587, 0.]) #[0.299, 0.587, 0.114])


def make_kernal(n):
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernal
    
def contact_detection(img):  #detect contact 
    
    
    im_gray = rgb2gray(img)
    ill_back = cv2.GaussianBlur(im_gray,(25,15),21)
    im_uniform = (im_gray - ill_back +50)*2+20 
    img = np.clip(im_uniform,0,255)
        
    im_canny = cv2.Canny(img.astype(np.uint8),70,100) #Being smaller than 70 does not seem too bad
    kernal1 = make_kernal(9)
    kernal2 = make_kernal(17)
    kernal3 = make_kernal(37)    
    img_d = cv2.dilate(im_canny, kernal1, iterations=1)
    
    img_e = cv2.erode(img_d, kernal1, iterations=1)
    
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    contact = cv2.dilate(img_ee, kernal3, iterations=1).astype(np.uint8)
    
    return contact
  







data_dir = loc_dir + 'data/{}_tracker/'.format(object_name)
data_ERIC =  glob(data_dir + 'encoding_*')
depths_paths =  glob(data_dir + 'pre_*')
depths_paths.sort()

model_NN = model.contrastive()
model_NN.load_state_dict(torch.load(loc_dir + '{}'.format(model_name)))

for i in data_ERIC:
    encoding = np.load(i)
    encodings += [encoding]

encodings = np.asarray(encodings)

encodings = torch.from_numpy(encodings).to('cuda')

#############################################################################################################


main_dir = '/home/mcube/data_brick/data_brick_daolin/'
'''
depths = np.load(main_dir + 'depths_40.npy')
trans = np.load(main_dir + 'trans_40.npy')
#rots = np.load('/home/mcube-daolin/tactile_localization/src/rots_v2.npy')
#poses = np.load('/home/mcube-daolin/tactile_localization/src/poses_v2.npy')
'''
size = 100
shape = 235
#aa = np.random.randint(0,shape,size)
#bb = np.random.randint(0,shape,size)
desired_dist = 0.0262  # TODO: be careful


MaskFilePath = main_dir +'calibration/' + 'mask.npy'
mask_black = np.load(MaskFilePath)
mask_black = cv2.resize(mask_black, (shape,shape))
mask_black = cv2.flip(mask_black,0)
#depths = np.multiply(depths, mask_black[:,:,0])

it_fake = 0
if __name__ == '__main__':
    
    while True:
        im_data = None
        while im_data is None:
            try: 
                combined = np.load(main_dir + 'test/combined.npy')
                print(combined.shape)
                im_size = int(combined.shape[1]/2)
                im_data = combined[:,im_size:]
                outImage = combined[:,:im_size]
            except:
                print('No im data')
        im = util.tensor2im(im_data)/170000;
        
        im[im > np.amax(im)*0.9] = 0.3
        im += desired_dist
        np.save(main_dir +'test/predicted_LS.npy', im)
               
        im_small = cv2.resize(im,(pixels,pixels))
        pre_depth = np.array(im_small[:,:,0])
        pre_depth = (pre_depth < max_val_depth).astype(np.int)
        pre_depth = np.multiply(pre_depth, mask_black[:,:,0])
        
        x = torch.rand(1, 3, pixels, pixels).cuda()
        x[0][0] = torch.from_numpy(pre_depth)
        x[0][1] = torch.from_numpy(pre_depth)
        x[0][2] = torch.from_numpy(pre_depth)

        encoding = model_NN.simple_forward(x).mean(2).mean(2).mean(0)
        encoding = encoding.unsqueeze(0)
        best = 0.
        argBest = 0

        it = 0
        #for x_val in self.train_loader:
        start = time.time()
        for i in range(int(len(encodings)/batch_size)):
            #a = self.model.cos(encoding, x_val)
            a = model_NN.cos(encoding, encodings[i*batch_size:min([(i+1)*batch_size, len(encodings)])])

            maxim, b = torch.max(a, 0)
            if (maxim > best):
                best = maxim
                argBest = it*batch_size + b
            it += 1
        print(time.time() - start)
        im_name = data_ERIC[int(argBest)]

        _, im_name = im_name.split('encoding')
        im_name, _ = im_name.split('.')

        _, ang, x, y = im_name.split('_')
        ang = int(ang)
        x = int(x)
        y = int(y)
        im_path = data_dir + 'pre_depth' + im_name + '.npz'
        pre_depth2 = np.load(im_path)['arr_0']
        pre_depth_v2 = np.load(im_path)['arr_0']
        pre_depth2 = (pre_depth2 < max_val_depth).astype(np.int)


        pd3 = np.reshape(np.matrix.repeat(pre_depth*255.0, 3), (pixels,pixels, 3))
        pd23 = np.reshape(np.matrix.repeat(pre_depth2*255.0, 3), (pixels,pixels, 3))
        all_3 =  np.concatenate([cv2.resize(outImage,(pixels,pixels)), pd3, pd23], axis=1)
        cv2.imwrite('b.png', all_3)
        
        np.save(main_dir +'test/im.npy', all_3); print('saved');
        trans = np.load(data_dir + 'transformation' + im_name + '.npy')

        pcd_local = get_world_pointcloud(pre_depth_v2, fx/2, fy/2)[0]
        pcd_local = np.dot(np.dot(pcd_local, np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T) - trans[:3,3], trans[:3,:3])
        
        
        np.save(main_dir + '../pred_pcd.npy', pcd_local)
        np.save(main_dir + '../pred_pos.npy', trans[:3,3])
        np.save(main_dir + '../pred_rot.npy', trans[:3,:3])
        
        
