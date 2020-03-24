import numpy as np
from os.path import join as pjoin

from fury import actor, window, ui



scene = window.Scene()
scene.background((1, 1, 1))

open_static = ui.TextBlock2D()
open_static.message = "Select template"

use_dir = ui.TextBlock2D()
use_dir.message = "Use directory"

out_dir = ui.TextBlock2D()
out_dir.message = "Output directory"

from pathlib import Path
home = str(Path.home())

# home = 'C:\\Users\\elef\\Data\\Ngn zebrafish VPA dose response experiment -20190922T000613Z-001\\'
# home += 'Ngn zebrafish VPA dose response experiment\\Control DMSO'

file_menu = ui.FileMenu2D(home, size=(1200, 400))

panel = ui.Panel2D(size=(1800, 420))


class ProcessingObject(object):

    ftemplate = None
    dname = None
    out_dname = None


po = ProcessingObject()


def open_static_callback(obj, event):
    po.ftemplate = pjoin(file_menu.current_directory,
                       file_menu.listbox.selected[0])
    open_static.message +=  ' ' + file_menu.listbox.selected[0]
    showm.render()

    print(po.ftemplate)


open_static.actor.AddObserver('LeftButtonPressEvent',
                              open_static_callback,
                              1.0)

def use_dir_callback(obj, event):
    po.dname = file_menu.current_directory
    use_dir.message += ' selected!'
    showm.render()
    print(po.dname)


use_dir.actor.AddObserver('LeftButtonPressEvent',
                          use_dir_callback,
                          1.0)


def out_dir_callback(obj, event):
    po.out_dname = file_menu.current_directory
    out_dir.message += ' selected!'
    showm.render()
    print(po.out_dname)


out_dir.actor.AddObserver('LeftButtonPressEvent',
                          out_dir_callback,
                          1.0)


showm = window.ShowManager(scene, size=(1980, 1080))

showm.initialize()
panel.add_element(file_menu, coords=(450, 10))
panel.add_element(open_static, coords=(20, 200))
panel.add_element(use_dir, coords=(20, 110))
panel.add_element(out_dir, coords=(20, 10))


scene.add(panel)
scene.add(actor.axes())

showm.render()
showm.start()


from zebrafish_alignment import imread, imwrite, mask_zebrafish, ssd_rotate_and_center, basename, glob, build_template, rgb2gray, zoom


if __name__ == "__main__":

    print(po.ftemplate)
    print(po.dname)
    print(po.out_dname)

    dname = po.dname
    zfs = glob(pjoin(dname, '*.tif'))

    print('Processing images from')
    for z in zfs:
        print(z)

    dout = po.out_dname

    print('Results will be saved in folder')
    print(dout)

    f1 = po.ftemplate

    z_size = 4
    threshold = 20
    data1 = imread(f1)

    # Specific to the new microscope for removing the label
    data1[900:, 800:] = 0

    data1 = rgb2gray(data1)

    data1 = mask_zebrafish(data1, threshold)

    # Careful this is tested only with DMSO_Image002_ch00.tif
    data1 = build_template(data1, -85)

    volume = np.zeros((data1.shape + (len(zfs) + 1,)))
    volume[..., 0] = data1

    imwrite(pjoin(dout, basename(f1) + '_template.png'), data1)

    rois = imread(pjoin(dout, 'rois.png'))


    data1 = zoom(data1, zoom=0.5, order=1)



    for i in range(1, len(zfs)):

        f2 = pjoin(dname, zfs[i])

        print('Processing ' + f2 + '..')

        data2 = imread(f2)

        data2[900:, 800:] = 0

        data2 = rgb2gray(data2)

        data2 = mask_zebrafish(data2, threshold)

        print(data1.shape, data2.shape)

        data2 = zoom(data2, zoom=0.5, order=1)

        data2_moved = ssd_rotate_and_center(data1, data2)

        data2_moved = zoom(data2_moved, zoom=2.0, order=1)

        print('Saving ' +  pjoin(dout, basename(zfs[i]) + '_aligned.png'))
        imwrite(pjoin(dout, basename(zfs[i]) + '_aligned.png'), data2_moved)

        volume[..., i] = data2_moved


    from show_tiffs import show_volume

    show_volume(volume)

    1/0

