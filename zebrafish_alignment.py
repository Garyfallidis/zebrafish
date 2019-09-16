"""
Zebrafish alignment

Eleftherios Garyfallidis
"""
import sys
import numpy as np
from glob import glob
from os.path import expanduser, join
from imageio import imread, imwrite
from dipy.viz import actor, window, ui

from dipy.align.imaffine import transform_centers_of_mass

from scipy.ndimage.interpolation import rotate, shift, affine_transform
from scipy.ndimage.measurements import center_of_mass
from os.path import basename


def rot_pt(img, pt, theta):
    # img[img<thresh]=0 #mask background
    c_in = np.array(pt)
    c_out = np.array(pt)
    transform = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta),np.cos(theta)]])
    offset = c_in - c_out.dot(transform)
    dst = affine_transform(
            img, transform.T, order=2,
            offset=offset, output_shape=img.shape, cval=0.0)
    return dst


def ssd_rotate_and_center(static, moving):

    # figure()
    # title('Before registration')
    # imshow(static + moving)

    # TODO I can try rotating the image around its center of mass and then
    # apply the c_of_mass transform

    ssds = []
    angles = range(0, 360)

    for theta in angles:

        movingn = rotate(moving, theta, reshape=False)
        c_of_mass = transform_centers_of_mass(static, None,
                                              movingn, None)
        moved = c_of_mass.transform(movingn)
        ssds.append(np.sum((moved - static) ** 2))

    best_ssd = np.argmin(ssds)
    print(np.min(ssds))
    print(angles[best_ssd])
    movingn = rotate(moving, angles[best_ssd], reshape=False)
    c_of_mass = transform_centers_of_mass(static, None,
                                          movingn, None)
    moved = c_of_mass.transform(movingn)

    return moved


def mask_zebrafish(data, thr):
    data[data < thr] = 0
    return data


#def debug_figure(data1, data2, data2_moved):
#
#    figure()
#
#    subplot(2, 2, 1)
#    imshow(data1)
#    title('fixed ' + basename(zfs[0]))
#
#    subplot(2, 2, 2)

#    imshow(data2)
#    title('moving ' + basename(zfs[i]))
#
#    subplot(2, 2, 3)
#    imshow(data2_moved)
#    title(basename(zfs[i] + '_registered'))
#
#    subplot(2, 2, 4)
#    imshow(data1 + data2_moved)
#    title('registered and fixed together')

def rotate_at_point(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]


def build_template(data1, rot_angle=10):
    # input should be already masked

    fine_template = rotate(data1, rot_angle, reshape=False)

    fts = fine_template.shape

    # notice that I am calculating center of mass for one figure
    cmass = center_of_mass(fine_template > 0)

    fine_template2 = shift(fine_template, [fts[0]/2 - cmass[0],
                           fts[1]/2 - cmass[1]])

    return fine_template2

wrng_msg = """
Wrong number of inputs'
Provide template image, directory of tif images to align, output folder'
For example:
python zebrafish_alignment.py static.tif DMSO out_dir
"""


if __name__ == '__main__':

    """ Provide template image, directory of tif images to align, output folder
    """

    if len(sys.argv == 4):
       print(wrng_msg)

    static = sys.argv[1]
    # moving = sys.argv[2]
    print(static)

    dname = sys.argv[2]
    print(dname)

    zfs = glob(join(dname, '*.tif'))

    print('Processing images from')
    for z in zfs:
        print(z)

    dout = sys.argv[3]

    print('Results will be saved in folder')
    print(dout)

    f1 = static  # join(dname, static)

    z_size = 4
    threshold = 300
    data1 = imread(f1)
    data1 = mask_zebrafish(data1, threshold)

    # imwrite(basename(zfs[0]), data1)
    #    data1_interp = np.interp(
    #            data1,
    #            [data1.min(), np.percentile(data1[data1 > 0], 95)],
    #            [0, 255])
    # imshow(data1_interp)

    for i in range(1, len(zfs)):

        f2 = join(dname, zfs[i])
        print('Processing ' + f2 + '..')

        data2 = imread(f2)
        data2 = mask_zebrafish(data2, threshold)

        data2_moved = ssd_rotate_and_center(data1, data2)

        print('Saving ' +  join(dout, basename(zfs[i]) + '_aligned.tiff'))
        imwrite(join(dout, basename(zfs[i]) + '_aligned.tiff'), data2_moved)



