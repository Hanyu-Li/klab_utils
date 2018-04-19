#!/usr/bin/env python3

from __future__ import print_function

import webbrowser
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import neuroglancer
import dxchange
import argparse
import os, signal
import re
from .reader import omni_read
from ast import literal_eval as make_tuple

def glance(viewer, image=None, labels=None, resolution=None):
    with viewer.txn() as s:
        s.voxel_size = resolution
        if image is not None:
            s.layers.append(
                name='image',
                layer=neuroglancer.LocalVolume(
                    data=image,
                    offset = (0,0,0),
                    voxel_size = s.voxel_size,
                ),
                shader="""
        void main() {
        emitRGB(vec3(toNormalized(getDataValue(0)),
                    toNormalized(getDataValue(1)),
                    toNormalized(getDataValue(2))));
        }
        """),
        if labels is not None:
            s.layers.append(
                name='labels',
                layer=neuroglancer.LocalVolume(
                    data=labels,
                    offset = (0,0,0),
                    voxel_size = s.voxel_size,
                ),
            )
    return viewer.get_viewer_url()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', default=None)
    parser.add_argument( '--labels', default=None)
    parser.add_argument( '--multi', default=False)
    parser.add_argument( '--begin', type=int, default=None)
    parser.add_argument( '--end', type=int, default=None)
    parser.add_argument( '--resolution', type=str, default='(600,600,600)')
    parser.add_argument( '--p', type=int, default=42000 )
    args = parser.parse_args()

    image = omni_read(args.image, args.begin, args.end)
    labels = omni_read(args.labels, args.begin, args.end)
    resolution = make_tuple(args.resolution)

    if image.dtype != np.uint8:
        print('not 8bit')
        image = image * 255
    if labels is not None:
        if not args.multi:
            # only a single object
            labels = np.uint32(np.nan_to_num(labels)>0)
        else:
            labels = np.uint32(np.nan_to_num(labels))
    neuroglancer.set_server_bind_address(bind_address='127.0.0.1', bind_port=args.p)
    viewer = neuroglancer.Viewer()
    def dummy():
        print('hello callback')
    #viewer.defer_callback(dummy)
    url = glance(viewer=viewer, image=image, labels=labels, resolution=resolution)
    print(url)
    webbrowser.open_new_tab(url)


    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()

if __name__ == '__main__':
    main()