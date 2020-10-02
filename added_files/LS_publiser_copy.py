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
import model

from torch.utils.data import Dataset, TensorDataset

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split



from glob import glob

encodings = []

main_dir = '/home/mcube/tactile_localization/'
data_dir = main_dir + 'data/{}_tracker/'.format(object_name)
data =  glob(data_dir + 'encoding_*')

for i in data:
    encoding = np.load(i)
    encodings += [encoding]

encodings = np.asarray(encodings)

encodings = torch.from_numpy(encodings).to('cuda')

#############################################################################################################


main_dir = '/home/mcube/data_brick/data_brick_daolin/'

depths = np.load(main_dir + 'depths_40.npy')
trans = np.load(main_dir + 'trans_40.npy')
#rots = np.load('/home/mcube-daolin/tactile_localization/src/rots_v2.npy')
#poses = np.load('/home/mcube-daolin/tactile_localization/src/poses_v2.npy')
size = 100
shape = 40
#aa = np.random.randint(0,shape,size)
#bb = np.random.randint(0,shape,size)
desired_dist = 0.0262  # TODO: be careful


MaskFilePath = main_dir +'calibration/' + 'mask.npy'
mask_black = np.load(MaskFilePath)
mask_black = cv2.resize(mask_black, (shape,shape))
mask_black = cv2.flip(mask_black,0)
depths = np.multiply(depths, mask_black[:,:,0])

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot = main_dir + 'test/' ## Added!!
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print(opt.preprocess)
    while True:
        init_t = time.time()
        dataset = None
        while dataset is None:
            outImage = None
            while outImage is None:
                try:
                    outImage = np.load(main_dir + 'test/calibrated_image.npy') # slow but 0.01
                    it = int(time.time()*100)%2500
                    #outImage = cv2.imread('/home/mcube/tactile_data/dec_12/gs/gs_{}.png'.format(it)) # slow but 0.01

                except:
                    pass
            cv2.imwrite(main_dir +'/test/test/img.png', np.concatenate([outImage, outImage*0], axis=1))
            dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
            if opt.eval:
                model.eval()
            for i, data in enumerate(dataset):

                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                for label, im_data in visuals.items():
                    if 'fake' in label:
                        im = util.tensor2im(im_data)/170000;
                        im[im > np.amax(im)*0.9] = 0.3
                        im += desired_dist
                        np.save(main_dir +'test/predicted_LS.npy', im)
                        plt.imshow(im); plt.savefig(main_dir +'test/predicted_LS.png')
                        im_small = cv2.resize(im,(shape,shape))

                        #error = np.sqrt(np.mean((depths[:,aa,bb]-np.array([im_small[aa,bb,0]]))**2,axis=1)) #slowest part by far (0.1s), otherwise: 0.0007s
                        #PIXEL ERROR: error = np.sqrt(np.mean(np.mean((np.subtract(depths,np.array([im_small[:,:,0]])))**2,axis=-1),axis=-1)) #slowest part by far (0.1s), otherwise: 0.0007s
                        # NN error
                        
                        it_min = np.argmin(error)
                        #break
                    if 'A' in label:
                        gs_im = util.tensor2im(im_data)

#            except:
#                pass
            #np.save('/home/mcube-daolin/predict_data/test/im.npy', im); print('saved')
            np.save(main_dir +'test/pred_im.npy', im[:,:,0]); print('saved')  ;
            cv2.imwrite(main_dir +'test/pred_im.png', im[:,:,0]);
            resized_depth = cv2.resize(depths[it_min],(im.shape[1],im.shape[0]))
            cv2.imwrite(main_dir + 'test/result.png', np.concatenate([im[:,:,0], resized_depth], axis=1)*750)
            cv2.imwrite(main_dir + 'test/results/result_{}.png'.format(it), np.concatenate([gs_im[:,:,1],np.concatenate([im[:,:,0], resized_depth], axis=1)*750], axis=1))
            np.save(main_dir +'test/im.npy', np.concatenate([im[:,:,0], cv2.resize(depths[it_min],(im.shape[1],im.shape[0]))], axis=1)); print('saved')  ;
            np.save(main_dir + '../pred_depth.npy', depths[it_min])
            np.save(main_dir + '../pred_pos.npy', trans[it_min,:3,3])
            np.save(main_dir + '../pred_rot.npy', trans[it_min,:3,:3])
            #plt.imshow(np.concatenate([im[:,:,0], depths[it_min]], axis=1)); plt.show()
