'''Converts a image stack to cloudvolume '''
from __future__ import print_function
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import neuroglancer
import glob
import cv2
import sys
import dxchange
import argparse
from pprint import pprint
import os, signal
import re
from ast import literal_eval as make_tuple
from .reader import check_stack_len,omni_read 
import json
from scipy.ndimage.interpolation import zoom
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

logger = logging.getLogger(__name__)
def _find_index(fname):
    res = re.search(r'([0-9]+).(.*)', fname)     
    return int(res.group(1))


def mpi_cloud_write(f_sublist, c_path, start_z, mip, factor, chunk_size, cv_args):
    ''' f_sublist is a List[List[str]], inner list size == z_batch'''
    z_chunk = chunk_size[2] # 64 
    #print('length: ', len(f_sublist))
    for fb in tqdm(f_sublist):
        loaded_vol = np.stack([cv2.imread(f, 0) for f in fb], axis=2)
        curr_z = _find_index(fb[0])
        actual_z = curr_z - start_z
        for m in range(mip):
            cv = CloudVolume(c_path, mip=m, **cv_args)
            step = np.array(factor)**m
            cv_z_start = actual_z // step[2]
            cv_z_size = loaded_vol.shape[2] // step[2]
            cv[:,:,cv_z_start:cv_z_start + cv_z_size] = loaded_vol[
                ::step[0], ::step[1], ::step[2]]
    pass
def divide_work(f_list, num_proc, z_batch, image_size, memory_limit):
    '''Decide the best way to distribute workload and z_batch size.
        min granularity is z_batch
    '''
    batched_f_list = np.split(np.asarray(f_list), np.arange(z_batch, len(f_list), z_batch))
    #merge_batch_N = memory_limit // (z_batch * image_size)
    f_sublists = np.array_split(batched_f_list, num_proc)
    # for f in f_sublists:
    #     if len(f) > merge_batch_N:
    #         f = np.concatenate(f)


    return f_sublists




def stack_to_cloudvolume(input_dir, output_dir, layer_type, data_type, mip,
    chunk_size, resolution, memory_limit=32):
    '''Converts a stack of images to cloudvolume.'''
    if mpi_rank == 0:
        f_list = glob.glob(os.path.join(input_dir, '*'))
        f_list.sort()
        os.makedirs(output_dir, exist_ok=True)
        im0 = cv2.imread(f_list[0], 0)
        Z = len(f_list)
        X,Y = im0.shape
        start_z = _find_index(f_list[0])
        print('start_z:', start_z)
        im_size = np.float32(sys.getsizeof(im0)) / 1073741824
        print('single_image_size: \t %.4f GB' % im_size)
        print('total_size: \t\t %.4f GB' % (im_size * Z))
        print('mpi_proc: \t\t %d' % mpi_size)
        print('total_limit: \t\t %.4f GB' % memory_limit)
        f_sublist = divide_work(f_list, mpi_size, chunk_size[2], im_size, memory_limit)
        c_path = 'file://' + os.path.join(output_dir, 'images')
        factor = [2,2,1]
        info = CloudVolume.create_new_info(
            num_channels=1, 
            layer_type=layer_type, 
            data_type=str(data_type), 
            encoding='raw',
            resolution=list(resolution),
            voxel_offset=np.array([0,0,0]),
            volume_size=[X,Y,Z],
            chunk_size=chunk_size,
            max_mip=mip,
            factor=factor
            )
        cv_args = dict(
            bounded=True, fill_missing=False, autocrop=False, 
            cache=False, compress_cache=None, cdn_cache=False, 
            progress=False, info=info, provenance=None, compress=None, 
            non_aligned_writes=False, parallel=4)
        cv = CloudVolume(c_path, mip=0, **cv_args)
        cv.commit_info()
        #pprint(cv.info)

    else:
        f_sublist = None
        c_path = None
        start_z = None
        cv_args = None
        factor = None
    f_sublist = mpi_comm.scatter(f_sublist, root=0)
    c_path = mpi_comm.bcast(c_path, root=0)
    start_z = mpi_comm.bcast(start_z, root=0)
    cv_args = mpi_comm.bcast(cv_args, root=0)
    factor = mpi_comm.bcast(factor, root=0)

    print('rank: %d, range: %s <-> %s' % (
        mpi_rank, f_sublist[0][0], f_sublist[-1][-1]))
    mpi_cloud_write(f_sublist, c_path, start_z, mip, factor, chunk_size, cv_args)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default=None)
    parser.add_argument('--label_dir', default=None)
    parser.add_argument('--output_dir', default='./precomputed')
    parser.add_argument('--multi', default=False)
    parser.add_argument('--resolution', type=str, default='10,10,10')
    parser.add_argument('--mip', type=int, default=3)
    parser.add_argument('--chunk_size', type=str, default='64,64,64')
    parser.add_argument('--z_step', type=int, default=None)
    parser.add_argument('--flip_axes', type=bool, default=False)
    parser.add_argument('--memory_limit', type=float, default=32)
    
    args = parser.parse_args()
    resolution = tuple(int(d) for d in args.resolution.split(','))
    chunk_size = tuple(int(d) for d in args.chunk_size.split(','))

    stack_to_cloudvolume(args.image_dir, args.output_dir, 
        layer_type='image', data_type='uint8', 
        mip=args.mip,
        chunk_size=chunk_size, 
        resolution=resolution,
        memory_limit=args.memory_limit)


if __name__ == '__main__':
    main()







