# -*- coding: utf-8 -*-
"""
Zebrafish-alignment-withoutdipy
"""
#Import Libraries
import numpy as np
from PIL import Image
from imageio import imread, imwrite
from glob import glob
from os.path import expanduser, join, basename
from scipy.ndimage import affine_transform, rotate, shift
from scipy import linalg, sin, cos
from scipy.optimize import minimize
import scipy.ndimage as ndimage
from dipy.align.imaffine import transform_centers_of_mass
import matplotlib.pyplot as plt
import time

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

"""
Show image
"""
def show_image(image,title=""):
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
    
"""
Plot two images
"""
def plot_two_images(static, moving, display_transformed = True):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(static)
    plt.title("Reference image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(moving)
    if display_transformed:
        plt.title("Transformed image")
    else:
        plt.title("Target image")
        
    plt.axis("off")

"""
overlay 2 images, on top of another
"""
def overlap_images(static, moving):
    show_image((static+moving)/2)
    

"""
Convert RGB image to gray
"""
def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, : , 1], rgb[: , :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

"""
Threshold image
"""
def threshold_image(im):
    im[im<np.mean(im)]=0
#     im[im<10]=0
    return im

"""
Read and standardize image
"""
def standardize_image(file):
    image = rgb2gray(imread(file))
    image[900:, 800:] = 0
    standard = threshold_image(image)
    
    return standard

"""
Get file name without file extension
"""
def basefile(file_path):
    file_name = basename(file_path)
#     index_of_dot = file_name.index('.')
#     file_name_without_extension = file_name[:index_of_dot]
    return file_name.rsplit('.', 1)[0] #file_name_without_extension

"""
Method to compute ssd between two arrays
"""
def ssd(arr1,arr2):
    """ Compute the sum squared difference metric """
    x = min(arr1.shape[0],arr2.shape[0])
    y = min(arr1.shape[1],arr2.shape[1])
    return np.sum((arr1[:x,:y]-arr2[:x,:y])**2)

# function to generate rotation matrix given 
def rot_matrix(ang):
    """
    Args:
        ang - angle of rotation in degrees
        
    Returns:
       A 2X2 array which rotation transformation matrix 
    """
    rad = np.deg2rad(np.asarray(ang).item()) #np.asarray(ang).item() is added to handle if ang is single value list
    m = np.array([[cos(rad), sin(rad)],
                     [-sin(rad), cos(rad)]])
    return m


# function to calculate offset point which determines rotation point of image
def cal_offset(ang):
    """
    Args:
        ang - angle of rotation in degrees 
        
    Returns:
       A offeset coordinates which is helpful in affine transformation
       (where the transformation needs to be applied)
    """
    stat_cm = ndimage.measurements.center_of_mass(np.array(static))
    return -((np.array(stat_cm)-np.array(stat_cm).dot(rot_matrix(ang))).dot(linalg.inv(rot_matrix(ang))))

# function which calculates the ssd for each rotation on moving image with static
# this function will be used in the optimizer part 2. of assignment
def cost_func(ang, stat_img, moving_img):
    """
    Args:
        ang - angle of rotation in degrees 
        stat_img - static or reference image
        moving_img - moving image which is same size as stat_img
        
    Returns:
       A cost i.e., SSD between static image and moving image rotated by angle = ang degrees
    """
    transformed_img = affine_transform(moving_img,rot_matrix(ang),offset=cal_offset(ang))
    cost = ssd(stat_img, transformed_img)
    return cost


## Main function
def align_zebrafish(static, moving):
    #align centre of mass
    c_of_mass = transform_centers_of_mass(static, None, moving, None)
    transformed = c_of_mass.transform(moving)

    moving = np.copy(transformed)
    
    # rotate and register
    best_ang = minimize(cost_func, 1, method = 'Powell', args = (static,moving), bounds=(-360,360))
    opt_ang = best_ang['x'].item()
    print("Best Angle", opt_ang)
    optimizer_transformed = affine_transform(moving,rot_matrix(opt_ang),offset=cal_offset(opt_ang))

    #plotting both static image and registered image side-by-side
#     plot_two_images(static, optimizer_transformed, True)
    
    #plotting overlap of static and transformed image
    plt.figure()
    overlap_images(static, optimizer_transformed)
    
    return optimizer_transformed   

if __name__ == '__main__':

    """ Provide template image, directory of tif images to align, output folder
    """
    start_time = time.time()
    
    static_file = "30_uM_VPA_Image015_ch00.tif"
    # dname = "Control_DMSO"
    dname = "30_uM_VPA"
    dout = "zebrafish_alignment_outputs"
    
    print(dname)
    
    zfs = glob(join(dname, '*.tif'))
    
    print('Processing images from')
    for z in zfs:
        print(z)
    
    print('Results will be saved in folder')
    print(dout)
        
        
    static = standardize_image(static_file)
    
    for i in range(len(zfs)):
        f2 = join(zfs[i])
        print('-------------------------------------------------')
        print('Processing ' + f2 + '..')
        moving = standardize_image(f2)
        transformed = align_zebrafish(static, moving)
        print('Saving ' +  join(dout, basefile(zfs[i]) + '_aligned.tif'))
        imwrite(join(dout, basefile(zfs[i]) + '_aligned.tif'), transformed)
    
    print("--- %s seconds ---" % (time.time() - start_time))
