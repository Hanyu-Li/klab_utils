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
import os, signal
import re
from ast import literal_eval as make_tuple
from .neuroglance_raw import omni_read 
import json
from cloudvolume import CloudVolume

logger = logging.getLogger(__name__)

def local_to_cloud(data, cloud_path, layer_type=None, resolution=None):
    '''currently support 
        layer_type: 'image' or 'segmentation'
        resolution: tuple of 3 '''
    if not os.path.exists(cloud_path):
        os.makedirs(cloud_path)

    info = CloudVolume.create_new_info(1, 
            layer_type=layer_type, 
            data_type=str(data.dtype), 
            encoding='raw',
            resolution=resolution,
            voxel_offset=(0,0,0),
            volume_size=data.shape
            )
    #print(info)
    #cloud_path = '/home/hanyu/disk_raid/KLab_Util_test/cloud_test/image' # Basic Example
    with open(os.path.join(cloud_path, 'info'), 'w') as f:
        json.dump(info, f)
    

    vol = CloudVolume('file://'+cloud_path, compress='') # Basic Example
    print(vol.volume_size)
    vol[:,:,:] = data

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', default=None)
    parser.add_argument( '--labels', default=None)
    parser.add_argument( '--precomputed', default=None)
    parser.add_argument( '--multi', type=bool, default=False)
    parser.add_argument( '--begin', type=int, default=None)
    parser.add_argument( '--end', type=int, default=None)
    parser.add_argument( '--resolution', type=str, default=None)
    
    parser.add_argument( '--p', type=int, default=42000 )
    args = parser.parse_args()
    image = omni_read(args.image, args.begin, args.end)
    labels = omni_read(args.labels, args.begin, args.end)

    resolution = make_tuple(args.resolution)

    if image is not None: 
        print(image.shape, image.dtype)
        image_cloud_path = os.path.join(args.precomputed, 'image')
        local_to_cloud(image, image_cloud_path, layer_type='image', resolution=resolution)


    if labels is not None: 
        if not args.multi:
            labels = np.uint32(np.nan_to_num(labels)>0)
        else:
            labels = np.uint32(np.nan_to_num(labels))
        print(labels.shape, labels.dtype)
        labels_cloud_path = os.path.join(args.precomputed, 'labels')
        local_to_cloud(labels, labels_cloud_path, layer_type='segmentation', resolution=resolution)
    



if __name__ == '__main__':
    main()







