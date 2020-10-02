###### Goal of this code ##############
#
#   Given a set of tactile images (png)   and depth maps (npy)
#       train different types of models
#######################################

import os
import helper_train

### FUNCTIONS NEEDED:
#
#  get dif/norm_diff from raw and pre_gs

# add canny into imput images

# add data augmentation --> save it?

# add mask? calibration? ufff

# DONE add extrachannels with pos

# DONE depth_from_gradients : compute depth from gradients

# compute gradients from depth

# compute mask from depth

# compute depth from canny --> ufff


###### INPUT  ##############
size_input = 470
input_types = ['raw', 'dif', 'norm_dif']
blue_channel = ['blue', 'canny']
data_aug = ['with_aug', 'no_aug']
mask_type = ['regular']
extra_pos_channels = ['yes', 'no']


###### DATA TYPE #########

DATA_TYPES = ['dec_3', 'dec_15', 'dec_18', 'dec_20', 'old', 'mid', 'dec_3_small', 'dec_15_small', 'dec_18_small']
rm_too_small = ['yes', 'no']


##### OUTPUT #####
output_types = ['depth','mask','gradients']  #make sure values depth for png are similar old and new
output_rendering = ['sim', 'anal']


#### ARCHITECTURES #####


data_types = ['NN', 'pix2pix']

if pix2pix:
    with_GANs = ['0_GAN',  '1_GAN']


#### OTHER PARAMS ####

number_datapoints = 2000
number_epochs = 40
#####

#### METRIQUES #####

# test_types = ['dec_12', 'dec_19'] #add others
# save worst for training and for testing

for with_GAN in with_GANs:
  for data_type in data_types:
    print( data_type, with_GAN)
    #command = 'python train.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_{0}_resnet_dec_18_{1} --dataset_mode tactile --model pix2pix --direction AtoB --niter 40 --niter_decay 1 --save_epoch_freq 20 --netG resnet_9blocks'.format(data_type, with_GAN)
    test_command = 'python test.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_{0}_resnet_dec_18_{1}  --model pix2pix --direction AtoB --ntest 100 --num_test 50 --netG resnet_9blocks'.format(data_type, with_GAN)

    #os.system(command)
    os.system(test_command)
