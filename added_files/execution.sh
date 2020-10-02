

#python train.py --dataroot /home/gridsan/cflc/claudia/object_train_pointrend/2500/  --name pix2pixtrain_train_train_test_pin --dataset_mode tactile --model pix2pix --direction AtoB  --save_epoch_freq 20 --netG resnet_9blocks


python test.py --dataroot /home/gridsan/cflc/claudia/px2px_data_pin/  --name pix2pixtest_train_train_test_pin  --model pix2pix --direction AtoB  --num_test 100 --netG resnet_9blocks --norm group --model test --no_dropout

#data root is where 'train' 'val' 'test' folders live
# name is just name, change it at will
