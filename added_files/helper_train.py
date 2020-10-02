import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
import cv2
import scipy
import scipy.io
import cPickle as pickle
import h5py
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import matplotlib.image as mpimg
import math
import scipy, scipy.fftpack
import yaml


def add_pos_to_input(img):
  dim = img.shape
  xvalues = np.array(range(dim[0])).astype('float32')/float(dim[0]) -0.5  # Normalized
  yvalues = np.array(range(dim[1])).astype('float32')/float(dim[1])  -0.5 # Normalized
  pos = np.stack((np.meshgrid(yvalues, xvalues)), axis = 2)
  
  img = np.concatenate((img, pos),axis = 2)
  return img



def plot_depth_map(depth_map, show=True, save=False, path='', img_number = '', top_view=False, palette=cm.BuPu):

    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    ## 3 lines below take no time
    X = np.arange(depth_map.shape[0], step=1)
    Y = np.arange(depth_map.shape[1], step=1)
    X, Y = np.meshgrid(X, Y)
    
    save_time_start = time.time()
    palette = cm.Reds
    surf = ax.plot_surface(X, Y, np.transpose(depth_map), rstride=1, cstride=1, cmap=palette, linewidth=0, antialiased=False)
    
    # 2 lines below take no time
    ax.set_zlim(0, 5); ax.view_init(elev=45., azim=5)
    if top_view: ax.view_init(elev=90., azim=0)
    ax.view_init(elev=90., azim=0)
    plt.axis("off")
    depth_map[depth_map < 1] = 0
    if save: 
        plt.savefig(path + "img_" + str(img_number) + ".png")
    if show: plt.show()
    
    '''
        fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    '''

def poisson_reconstruct(grady, gradx):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # We set boundary conditioons
    # print gradx.shape
    boundarysrc = np.zeros(gradx.shape)

    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0;

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    # print f.shape
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt
    positive_mask = (result > 0).astype(np.float32)
    result = result*positive_mask
    return result


def depth_from_gradients(gx, gy)
    depth = poisson_reconstruct(gx, gy)
    return depth

def gradients_from_depth(depth):
  '''
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])*4
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])*4
    
    edges_x = cv2.filter2D(depth,cv2.CV_8U,kernelx)
    edges_y = cv2.filter2D(depth,cv2.CV_8U,kernely)
  '''
    vgrad = np.gradient(varray)
    return vgrad[0], vgrad[1]
