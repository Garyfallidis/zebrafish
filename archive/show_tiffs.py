# -*- coding: utf-8 -*-
"""
Zebrafish alignment

Eleftherios Garyfallidis
"""

import sys
import numpy as np
from glob import glob
from os.path import expanduser, join
from imageio import imread, imwrite
from dipy.viz import actor, window, ui, app

from dipy.align.imaffine import transform_centers_of_mass

from scipy.ndimage.interpolation import rotate
from os.path import basename
from zebrafish_alignment import mask_zebrafish



def show_volume(vol, affine=np.eye(4), opacity=1.):

    app.horizon(images=[(vol, affine)])




def show_volume2(vol, affine=np.eye(4), opacity=1.):
    import vtk
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

        print('Value of voxel [%i, %i, %i]=%s'
              % (I, J, K, str(np.round(vol[I, J, K]))))

        # print("Picking 3d position")
        # print(obj.GetPickPosition())

    show_m.picker.AddObserver("EndPickEvent", annotate_pick)
    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()


if __name__ == '__main__':

    static = sys.argv[1]
    # moving = sys.argv[2]
    print(static)

    # home = expanduser('~')
    # dname = join(home, 'Data', 'zebrafish')

    movedfs = sys.argv[2]
    #print(dname)

    zfs = glob(movedfs)

    print(zfs)

    f1 = static # join(dname, static)

    z_size = 4
    threshold = 300
    data1 = imread(f1)
    data1 = mask_zebrafish(data1, threshold)

    data1_interp = np.interp(
            data1,
            [data1.min(), np.percentile(data1[data1 > 0], 95)],
            [0, 255])
    # imshow(data1_interp)

    volume = np.zeros((data1.shape + (len(zfs) + 1,)))
    volume[..., 0] = data1

    for i in range(1, len(zfs) + 1):

        f2 = join(zfs[i - 1])
        print('Processing ' + f2 + '..')

        data2 = imread(f2)

        volume[..., i] = data2

    show_volume(volume)



