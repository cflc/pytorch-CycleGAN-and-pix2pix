

import os


data_types = ['both', 'dif', 'norm_dif']
with_GANs = ['0_GAN']

for with_GAN in with_GANs:
  for data_type in data_types:
    print( data_type, with_GAN)
    
    command = 'python train.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_no_aug_{0}_resnet_dec_18_{1} --dataset_mode tactile --model pix2pix --direction AtoB --niter 40 --niter_decay 1 --save_epoch_freq 20 --netG resnet_9blocks --preprocess none'.format(data_type, with_GAN)
    test_command = 'python test.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_no_aug_{0}_resnet_dec_18_{1}  --model pix2pix --direction AtoB --ntest 100 --num_test 50 --netG resnet_9blocks --preprocess none'.format(data_type, with_GAN)

    os.system(command)
    os.system(test_command)
    
    print( data_type, with_GAN)
    command = 'python train.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_group_{0}_resnet_dec_18_{1} --dataset_mode tactile --model pix2pix --direction AtoB --niter 40 --niter_decay 1 --save_epoch_freq 20 --netG resnet_9blocks --norm group'.format(data_type, with_GAN)
    test_command = 'python test.py --dataroot /home/mcube/tactile_data/dec_18/{0} --name tactile_group_{0}_resnet_dec_18_{1}  --model pix2pix --direction AtoB --ntest 100 --num_test 50 --netG resnet_9blocks --preprocess none'.format(data_type, with_GAN)

    os.system(command)
    os.system(test_command)
