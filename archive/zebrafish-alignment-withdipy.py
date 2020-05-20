# -*- coding: utf-8 -*-
"""
Zebrafish-alignment-withdipy
"""

import numpy as np
from PIL import Image
import imageio
from glob import glob
from os.path import expanduser, join
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 SSDMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,RotationTransform2D,
                                   AffineTransform2D)
import scipy.ndimage 
import matplotlib.pyplot as plt

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

######## Helper Functions ###############
"""
Read image
"""
# def read_image(file):
#     image = imageio.imread(file)
#     return image

"""
Show image
"""
# def show_image(image,title=""):
#     plt.imshow(image)
#     plt.title(title)
#     plt.axis("off")
#     plt.show()
    
    
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
    return im

"""
Read and standardize image
"""
def standardize_image(file):
    image = rgb2gray(imageio.imread(file))
    image[900:, 800:] = 0
    standard = threshold_image(image)
    
    return standard

if __name__ == '__main__':
    
    """ Provide template image, directory of tif images to align, output folder
    """
    
    static_file = "30_uM_VPA_Image015_ch00.tif"
    dname = "30_uM_VPA"
    dout = "zebrafish_alignment_outputs"
    # print(dname)
    
    zfs = glob(join(dname, '*.tif'))
    
    # print('Processing images from')
    # for z in zfs:
    #     print(z)
    
    static = standardize_image(static_file)
    moving = standardize_image(zfs[1])
    
    
    c_of_mass = transform_centers_of_mass(static, None, moving, None)
    print("Images with center of mass transformed")
    transformed = c_of_mass.transform(moving)
    
    # level_iters = [1000, 1500, 100]
    level_iters = [100000]
    # sigmas = [3.0, 1.0, 0.0]
    sigmas = [0.0]
    # factors = [4, 2, 1]
    factors = [1]
    
    # initialize the metric
    metric = SSDMetric()
    affine_transform = RotationTransform2D()
    
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    
    params0 = [0]
    starting_affine = c_of_mass.affine
    current_affine, params, fopt = affreg.optimize(static, moving, affine_transform, params0,
                                  None, None,
                                  starting_affine=None,ret_metric = True)
    
    transformed2 = current_affine.transform(moving)
    transformed3 = scipy.ndimage.interpolation.rotate(moving,params.item(),reshape=False)
    
    plot_two_images(static, transformed,"Centre of Mass Transformed")
    plot_two_images(static, transformed2,"Centre of Mass Transformed")
    plot_two_images(static, transformed3,"Centre of Mass Transformed")