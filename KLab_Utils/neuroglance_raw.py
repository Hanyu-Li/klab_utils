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

def check_stack_len(f_name):
    matches = re.match(r'^.*/.*(?P<index>[0-9]+).(tif*)$',f_name)
    if matches: 
        #index = int(matches['index'])
        
        #print(match.groups())
        S = int(matches.group(1))
        L = len(os.listdir(os.path.dirname(f_name)))
        #L = len(glob.glob(os.path.dirname(f_name)))
        return (S, L)
    else:
        return (0,1)
def omni_read(f_input, begin=None, end=None):
    '''support tiff, tiff stack, hdf5'''
    if not f_input: return None
    matches = re.match(r'^(?P<dirname>.*)/(?P<fname>.*)\.(?P<ext>[^:]*)($|:(?P<dataset>.*$))', f_input)
    print(matches.groupdict())
    if matches['ext'] == 'tif' or matches['ext'] == 'tiff':
        if begin is not None and end is not None:
            data = dxchange.read_tiff_stack(f_input, ind=range(begin,end))
        else:
            print(f_input)
            S,L = check_stack_len(f_input)
            print(S,L)
            if L > 1:
                data = dxchange.read_tiff_stack(f_input, ind=range(S,S+L))
            else:
                data = dxchange.read_tiff(f_input)
    elif matches['ext'] == 'h5' or matches['ext'] == 'hdf5':
        tokens = f_input.split(':')
        dataset_name = tokens[1]
        f_input = h5py.File(tokens[0], 'r')
        data = np.asarray(f_input[tokens[1]])
    else:
        print('not implemented file type')
    return data
def glance(viewer, image=None, labels=None):
    with viewer.txn() as s:
        s.voxel_size = [600, 600, 600]
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
    parser.add_argument( '--multi', type=bool, default=False)
    parser.add_argument( '--begin', type=int, default=None)
    parser.add_argument( '--end', type=int, default=None)
    parser.add_argument( '--p', type=int, default=42000 )
    args = parser.parse_args()

    image = omni_read(args.image, args.begin, args.end)
    labels = omni_read(args.labels, args.begin, args.end)

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
    url = glance(viewer=viewer, image=image, labels=labels)
    print(url)
    webbrowser.open_new(url)


    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()

if __name__ == '__main__':
    main()