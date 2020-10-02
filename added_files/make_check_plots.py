from matplotlib import pyplot as plt
import numpy as np

name = 'tactile_dec_3_1_GAN_2k'
train = np.load('plots/train_error_{}.npy'.format(name))
test = np.load('plots/test_error_{}.npy'.format(name))

plt.plot(train, label='train')
plt.plot(test, label='test')
plt.legend()
plt.savefig('plots/train_vs_test_{}.png'.format(name))
    
