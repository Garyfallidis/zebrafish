# -*- coding: utf-8 -*-
"""
Zebrafish-group-iou-comparison-with-control
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
import sys
import matplotlib.ticker as mticker
from multiprocessing import Pool

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

"""
Show image
"""
def show_image(image,title=""):
    plt.figure()
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
overlay 2 images in different channels
"""
def overlap_channels(static, moving):
    # show_image(np.dstack([static,moving,np.zeros(static.shape)]))
    return np.dstack([static,moving,np.zeros(static.shape)])
    

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
    #mean as threshold
#     im[im<np.mean(im)]=0

    #otsu threshold
#     val = filters.threshold_otsu(im) #otsu_thresholding
#     im[im<val]=0
    
    #fixed threshold
    im[im<10.5]=0
    return im

"""
Intersection over Union to evaluate registration
"""
def iou(image1, image2):
    img1 = np.copy(image1)
    img1[img1<10.5] = 0
    img1[img1>=10.5] = 1
    
    img2 = np.copy(image2)
    img2[img2<10.5] = 0
    img2[img2>=10.5] = 1
    
    intersection = np.sum(img1 * img2)
    
    union = np.sum(img1+img2) - intersection
 
    
    
    return intersection / union

"""
Read and standardize image
"""
def standardize_image(file):
    im = imread(file)
    
    if len(im.shape) > 2:
        image = rgb2gray(im)
    else:
        image = im
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
def rot_matrix(ang, scale):
    """
    Args:
        ang - angle of rotation in degrees
        
    Returns:
       A 2X2 array which rotation transformation matrix 
    """
    rad = np.deg2rad(np.asarray(ang).item()) #np.asarray(ang).item() is added to handle if ang is single value list
    m = np.array([[scale*cos(rad), sin(rad)],
                     [-sin(rad), scale*cos(rad)]])
    return m


# function to calculate offset point which determines rotation point of image
def cal_offset(ang, scale, static):
    """
    Args:
        ang - angle of rotation in degrees 
        
    Returns:
       A offeset coordinates which is helpful in affine transformation
       (where the transformation needs to be applied)
    """
    stat_cm = ndimage.measurements.center_of_mass(np.array(static))
    return -((np.array(stat_cm)-np.array(stat_cm).dot(rot_matrix(ang, scale))).dot(linalg.inv(rot_matrix(ang, scale))))

# function which calculates the ssd for each rotation on moving image with static
# this function will be used in the optimizer part 2. of assignment
def cost_func(params, stat_img, moving_img):
    """
    Args:
        ang - angle of rotation in degrees 
        stat_img - static or reference image
        moving_img - moving image which is same size as stat_img
        
    Returns:
       A cost i.e., SSD between static image and moving image rotated by angle = ang degrees
    """
    ang, scale = params
    transformed_img = affine_transform(moving_img,rot_matrix(ang, scale),offset=cal_offset(ang, scale, stat_img))
    cost = ssd(stat_img, transformed_img)
    return cost

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# initial estimate of angle of rotation using pricipal eigen vector
def initial_estimate(img):

    x = np.nonzero(img)
    x = np.vstack(x).T
    x = x - x.mean(axis=0) 
    covx = np.dot(x.T, x) / x.shape[0]
    eigen_values, eigen_vectors = np.linalg.eigh(covx)

    return 180-np.rad2deg(np.arctan(eigen_vectors[1][1]/eigen_vectors[1][0]))


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

def optimal_align(static, moving):
    #align centre of mass
    c_of_mass = transform_centers_of_mass(static, None, moving, None)
    transformed = c_of_mass.transform(moving)

    moving = np.copy(transformed)
    
    # initial_angle = initial_estimate(moving)
    initial = [0, 1] #initial_estimate(moving)
    # rotate and register

    best_ang = minimize(cost_func, initial, method = 'Powell', args = (static,moving), bounds=((-180,180), (0.7, 1.2)))
    return best_ang, moving

def align_zebrafish2(static, moving):
    
    best_ang1, mov1 = optimal_align(static, moving)
    best_ang2, mov2 = optimal_align(static, np.fliplr(moving))
    
    cost1 = cost_func(best_ang1['x'], static,mov1)
    cost2 = cost_func(best_ang2['x'], static,mov2)
    
    if cost1 < cost2:
        opt_ang = best_ang1['x']
        print("Best Angle", opt_ang)
        mov = mov1
    else:
        opt_ang = best_ang2['x']
        print("Flip and Best Angle", opt_ang)
        mov = mov2
    optimizer_transformed = affine_transform(mov,rot_matrix(opt_ang[0], opt_ang[1]),offset=cal_offset(opt_ang[0], opt_ang[1], static))

    #plotting both static image and registered image side-by-side
#     plot_two_images(static, optimizer_transformed, True)
    
    
    #plotting overlap of static and transformed image
#     overlap_images(static, optimizer_transformed)
    
    #plotting overlap of static and transformed image
    overlap = overlap_channels(static, optimizer_transformed)
    
    return optimizer_transformed, overlap

def group_alignment(dirname):
    ious = []
    static = standardize_image('archive/template.tif')
    zfs = glob(join(dirname, '*.tif'))
    for i in range(len(zfs)):
        print("Started Registration of images in:", dirname)
        f1 = join(zfs[i])
        print('-------------------------------------------------')
        print('Processing ' + f1 + '..')
        moving = standardize_image(f1)
        transformed, _ = align_zebrafish2(static, moving)
        ious.append(iou(static,transformed))
        
    return ious

def reg_func(im_file):
    static = standardize_image('archive/template.tif')
    moving = standardize_image(im_file)
    transformed, _ = align_zebrafish2(static, moving)
    
    return iou(static,transformed)


def main(dname1, dname2):
    
    start_time = time.time()
    # control_ious = group_alignment(dname1)
    # subject_ious = group_alignment(dname2)
    
    p1 = Pool()
    print("Started Control Registration")
    control_ious = p1.map(reg_func, glob(join(dname1, '*.tif')))
    p1.close()
    p1.join()
    print("Completed Control Registration")
    
    p2 = Pool()
    print("Started Subject Registration")
    subject_ious = p2.map(reg_func, glob(join(dname2, '*.tif')))
    p2.close()
    p2.join()
    print("Completed Subject Registration")
    
    x = [i+1 for i, val in enumerate(control_ious)]
    plt.figure()
    plt.plot(x, control_ious, label="Control")
    plt.plot(x, subject_ious, label="Subject")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('Image Number')
    plt.ylabel('Intersection Over Union')
    plt.legend()
    plt.savefig('iou_plot.png')
    # plt.show()
    
    print("--- Total time taken is: %s minutes ---" % ((time.time() - start_time)/60))
    
    
if __name__ == '__main__':

    """ Provide template image, directory of tif images to align, output folder
    """
    start_time = time.time()
    
    
    # dname1 = "Control_DMSO"
    # dname2 = "30_uM_VPA"
    
    # main(dname1, dname2)
    
    main(sys.argv[1], sys.argv[2])
    
    print("--- %s seconds ---" % (time.time() - start_time))
