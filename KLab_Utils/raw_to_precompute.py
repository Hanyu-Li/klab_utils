#!/usr/env/bin python3
from __future__ import print_function
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import neuroglancer
import dxchange
import argparse
from pprint import pprint
import os, signal
import re
from ast import literal_eval as make_tuple
from .neuroglance_raw import omni_read 
import json
from scipy.ndimage.interpolation import zoom
from cloudvolume import CloudVolume

logger = logging.getLogger(__name__)
def build_pyramid_info(info, scale_up_to=3):
    if scale_up_to == 0: return info
    scale_0 = info['scales'][0]
    new_resolution = scale_0['resolution']
    new_size = scale_0['size']
    for i in range(1,scale_up_to):
        new_resolution = [r*2 for r in new_resolution]
        new_size = [s//2 for s in new_size]
        new_scale = {
            "encoding": scale_0['encoding'],
            "chunk_sizes": scale_0['chunk_sizes'],
            "key": "_".join(map(str, new_resolution)),
            "resolution": list(map(int, new_resolution)),
            "voxel_offset": list(map(int, scale_0['voxel_offset'])),
            "size": list(map(int, new_size))
        }



        info['scales'].append(new_scale)
    pprint(info)
    return info

def local_to_cloud(data, cloud_path, layer_type=None, resolution=None, scale=0):
    '''currently support 
        layer_type: 'image' or 'segmentation'
        resolution: tuple of 3 '''
    if not os.path.exists(cloud_path):
        os.makedirs(cloud_path)

    info = CloudVolume.create_new_info(1, 
            layer_type=layer_type, 
            data_type=str(data.dtype), 
            encoding='raw',
            resolution=list(resolution),
            voxel_offset=(0,0,0),
            volume_size=data.shape
            )

    info = build_pyramid_info(info, scale)
    with open(os.path.join(cloud_path, 'info'), 'w') as f:
        json.dump(info, f)

    for i in range(0,scale):    
        vol = CloudVolume('file://'+cloud_path, mip=i,compress='') # Basic Example
        if i > 0:
            data = zoom(data, 0.5)
            z,x,y = vol.volume_size
            data = data[0:z, 0:x, 0:y]
        print(vol.volume_size, data.shape)
        vol[:,:,:] = data



    #vol = CloudVolume('file://'+cloud_path, compress='') # Basic Example
    #print(vol.volume_size)
    #vol[:,:,:] = data

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', default=None)
    parser.add_argument( '--labels', default=None)
    parser.add_argument( '--precomputed', default=None)
    parser.add_argument( '--multi', type=bool, default=False)
    parser.add_argument( '--begin', type=int, default=None)
    parser.add_argument( '--end', type=int, default=None)
    parser.add_argument( '--resolution', type=str, default='(10,10,10)')
    parser.add_argument( '--scale', type=int, default=0)
    
    args = parser.parse_args()
    image = omni_read(args.image, args.begin, args.end)
    labels = omni_read(args.labels, args.begin, args.end)

    resolution = make_tuple(args.resolution)

    if image is not None: 
        print(image.shape, image.dtype)
        image_cloud_path = os.path.join(args.precomputed, 'image')
        local_to_cloud(image, image_cloud_path, layer_type='image', resolution=resolution, scale=args.scale)


    if labels is not None: 
        if not args.multi:
            labels = np.uint32(np.nan_to_num(labels)>0)
        else:
            labels = np.uint32(np.nan_to_num(labels))
        print(labels.shape, labels.dtype)
        labels_cloud_path = os.path.join(args.precomputed, 'labels')
        local_to_cloud(labels, labels_cloud_path, layer_type='segmentation', resolution=resolution, scale=args.scale)
    



if __name__ == '__main__':
    main()







