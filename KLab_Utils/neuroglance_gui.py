from __future__ import print_function

import webbrowser
import numpy as np
import sys
import h5py
import neuroglancer
import dxchange
import argparse
import os
import re

def check_stack_len(f_name):
    match = re.search(r'([0-9]+).(tif*)',f_name)
    if match is not None:
        #print(match.groups())
        S = int(match.group(1))
        L = len(os.listdir(os.path.dirname(f_name)))
        return (S, L)
    else:
        return (0,1)

def glance(viewer, raw=None, labels=None):
    with viewer.txn() as s:
        s.voxel_size = [600, 600, 600]
        if raw is not None:
            s.layers.append(
                name='image',
                layer=neuroglancer.LocalVolume(
                    data=raw,
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw',
        default=None
    )
    parser.add_argument(
        '--labels',
        default=None
    )
    parser.add_argument(
        '--multi',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--begin',
        type=int,
        default=None
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None
    )
    args = parser.parse_args()
    raw = None
    labels = None


    if args.raw is not None:
        if args.begin is not None and args.end is not None:
            S,L = args.begin, args.end
        else:
            S,L = check_stack_len(args.raw)
        print(S,L)
        if L > 1:
            raw = dxchange.read_tiff_stack(args.raw, ind=range(S,S+L))
        else:

            if args.raw.endswith('.tif') or args.raw.endswith('tiff'):
                raw = dxchange.read_tiff(args.raw)
            else: 
                tokens = args.raw.split(':')
                if tokens[0].endswith('.h5') or tokens[0].endswith('.hdf5'):
                    dataset_name = tokens[1]
                    f_input = h5py.File(tokens[0], 'r')
                    raw = f_input[tokens[1]]
    
    if args.labels is not None:
        if args.begin is not None and args.end is not None:
            S,L = args.begin, args.end
        else:
            S,L = check_stack_len(args.raw)
        if L > 1:
            labels = dxchange.read_tiff_stack(args.labels, ind=range(S, S+L))
        else:
            if args.labels.endswith('.h5') or args.labels.endswith('.hdf5'):
                f_labels = h5py.File(args.labels, 'r')
                labels = f_labels['stack']
            else:
                labels = dxchange.read_tiff(args.labels)
        print(labels.shape)
        if not args.multi:
            # only a single object
            labels = np.uint32(np.nan_to_num(labels)>0)
        else:
            labels = np.nan_to_num(labels)
        #labels = np.uint32(labels>128)
    viewer = neuroglancer.Viewer()
    def dummy():
        print('hello callback')
    viewer.defer_callback(dummy)
    url = glance(viewer=viewer, raw=raw, labels=labels)
    print(url)
    webbrowser.open_new(url)

