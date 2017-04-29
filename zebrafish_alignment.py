# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from os.path import expanduser, join
from scipy.misc import imread
import vtk
from dipy.viz import actor, window, widget
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D)

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
from dipy.segment.tissue import TissueClassifierHMRF
from scipy.ndimage.interpolation import affine_transform, rotate
from os.path import basename


def show_volume(vol, affine=np.eye(4), opacity=1.):

    ren = window.Renderer()

    shape = vol.shape

    image_actor = actor.slicer(vol, affine)

    slicer_opacity = opacity  # .6
    image_actor.opacity(slicer_opacity)

    ren.add(image_actor)

    show_m = window.ShowManager(ren, size=(800, 700))
    show_m.picker = vtk.vtkCellPicker()
    show_m.picker.SetTolerance(0.002)

    show_m.initialize()
    show_m.iren.SetPicker(show_m.picker)
    show_m.picker.Pick(10, 10, 0, ren)

    def change_slice(obj, event):
        z = int(np.round(obj.get_value()))
        image_actor.display_extent(0, shape[0] - 1,
                                   0, shape[1] - 1, z, z)
        ren.reset_clipping_range()

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=change_slice,
                           min_value=0,
                           max_value=shape[2] - 1,
                           value=shape[2] / 2,
                           label="Move slice",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            size = obj.GetSize()

    def annotate_pick(obj, event):
        I, J, K = obj.GetCellIJK()

        print('Value of voxel [%i, %i, %i]=%s' % (I, J, K, str(np.round(vol[I, J, K]))))
        # print("Picking 3d position")
        # print(obj.GetPickPosition())

    show_m.picker.AddObserver("EndPickEvent", annotate_pick)
    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()


def ssd_rotate_and_center(static, moving):

    figure()
    title('Before registration')
    imshow(static + moving)

    # TODO I can try rotating the image around its center of mass and then
    # apply the c_of_mass transform

    ssds = []
    angles = range(0, 180)

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


def affine_registration_2d(static, moving,
                           static_grid2world,
                           moving_grid2world,
                           level_iters=[100],
                           sigmas=[1.],
                           factors=[1]):

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    transformed = c_of_mass.transform(moving)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform2D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    transform = RigidTransform2D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigid.transform(moving)

    return transformed

    """
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    moved = affine.transform(moving)

    return moved
    """

def mask_zebrafish(data, thr):
    data[data < thr] = 0
    return data


home = expanduser('~')
dname = join(home, 'Data', 'zebrafish')

from glob import glob

zfs = glob(join(dname, '*.tif'))

f1 = join(dname, zfs[0])

z_size = 4
threshold = 300
data1 = imread(f1)
data1 = mask_zebrafish(data1, threshold)


for i in range(1, len(zfs)):

    f2 = join(dname, zfs[i])

    data2 = imread(f2)
    data2 = mask_zebrafish(data2, threshold)

    data2_moved = ssd_rotate_and_center(data1, data2)

    figure()

    subplot(2, 2, 1)
    imshow(data1)
    title('fixed ' + basename(zfs[0]))

    subplot(2, 2, 2)
    imshow(data2)
    title('moving ' + basename(zfs[i]))

    subplot(2, 2, 3)
    imshow(data2_moved)
    title(basename(zfs[i] + '_registered'))

    subplot(2, 2, 4)
    imshow(data1 + data2_moved)
    title('registered and fixed together')

    savefig(str(i) + '.png')
