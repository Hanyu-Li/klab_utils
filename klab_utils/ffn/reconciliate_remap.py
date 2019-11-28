'''First Stage in reconciliate of ffn inference sub volumes.

1. For each seg-*.npz, load subvol, make label contiguous
2. Find max id for each subvol and set offset for each subvol
3. Update each subvol with offset
4. Write to cloudvolume

'''
import cloudvolume
import re
import time
import numpy as np
from pprint import pprint
import argparse
import fastremap
from cloudvolume.lib import Bbox
from ffn.inference.segmentation import make_labels_contiguous
from ffn.inference.segmentation import clear_dust, make_labels_contiguous, clean_up
from ffn.inference import storage
# Agglomerate from seg folders and output to cloud volume 
# from klab_utils.ffn.export_inference import load_inference, get_zyx
import logging
import glob
import os
import re
from ffn.utils.bounding_box import BoundingBox
from cloudvolume.lib import Bbox
from cloudvolume import CloudVolume
from tqdm import tqdm
from ffn.utils import bounding_box
from ffn.utils.bounding_box import intersection
import itertools
import networkx as nx
import json
import pickle
import sys

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def get_zyx(fname):
  xyz = tuple([int(i) for i in re.search('seg-(\d+)_(\d+)_(\d+)\.npz', fname).groups()])
  zyx = (xyz[2], xyz[1], xyz[0])
  return zyx

def unify_ids(seg_map):
  keys = list(seg_map.keys())
  keys.sort()
  global_max = 0
  for k in keys:
    seg_map[k]['global_offset'] = global_max
    global_max += seg_map[k]['local_max']


def merge_dict(a, b, datatype):
  a.update(b)
  return a

def load_inference(input_dir):
  f = glob.glob(os.path.join(input_dir, '**/*.npz'), recursive=True)
  # f = [input_dir]
  # print('>>', f)
  #f = glob.glob(os.path.join(input_dir, '*.npz'))
  zyx = get_zyx(f[0])
  seg, _ = storage.load_segmentation(input_dir, zyx)
  return seg, np.array(zyx)
def get_seg_map(input_dir, output_dir, resolution, chunk_size, sub_indices, post_clean_up=False):
  if not sub_indices.size:
    return {}
  get_name = lambda s: re.sub('seg-', 'precomputed-', s)
  seg_list = glob.glob(os.path.join(input_dir, 'seg*'))
  seg_list.sort()
  
  seg_map = dict()
  pbar = tqdm(sub_indices, desc='Generating seg map')
  for i in pbar:
    seg_path = seg_list[i]
    if not len(glob.glob(os.path.join(seg_path, '**/*.npz'), recursive=True)):
      continue
    seg, offset_zyx = load_inference(seg_path)
    offset_xyz = offset_zyx[::-1]
    size_xyz = seg.shape[::-1]

    bbox = Bbox(a=offset_xyz, b=offset_xyz+size_xyz)
    seg = np.transpose(seg, [2, 1, 0])
    
    # make labels contiguous and unique globally but keep 0 intact
    min_particle = 1000
    if post_clean_up:
      clean_up(seg, min_particle) 
     
    seg = np.uint32(seg)
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    local_max = np.max(seg)
    
    precomputed_path = os.path.join(
      output_dir, get_name(os.path.basename(seg_path)))
    seg_map[i] = {
      'bbox': bbox, 
      'input': os.path.abspath(seg_path), 
      'output': os.path.abspath(precomputed_path), 
      'resolution': resolution,
      'chunk_size': chunk_size,
      'local_max': local_max}
  return seg_map

def prepare_precomputed(precomputed_path, offset, size, resolution, chunk_size, factor=(2,2,1), dtype='uint32'):
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=True)
  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='segmentation',
    data_type=dtype,
    # encoding='compressed_segmentation',
    encoding='raw',
    resolution=list(resolution),
    voxel_offset=np.array(offset),
    volume_size=np.array(size),
    chunk_size=chunk_size,
    max_mip=0,
    factor=factor,
    )
  cv = CloudVolume('file://'+precomputed_path, mip=0, info=info, **cv_args)
  cv.commit_info()
  return cv

def update_ids_and_write(seg_map, sub_indices):
  pbar = tqdm(sub_indices, desc='Update ids to be globally unique')
  for k in pbar:
    s = seg_map[k]['input']
    seg, offset_zyx = load_inference(s)
    offset_xyz = offset_zyx[::-1]
    size_xyz = seg.shape[::-1]

    bbox = seg_map[k]['bbox']
    seg = np.transpose(seg, [2, 1, 0])
    # # make labels contiguous and unique globally but keep 0 intact
    seg = np.uint32(seg)
    zeros = seg==0
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    # local_max = np.max(seg)
    seg += seg_map[k]['global_offset']
    seg[zeros] = 0

    # convert to precomputed
    precomputed_path = seg_map[k]['output']
    #seg_map[i] = (bbox, precomputed_path)
    resolution = seg_map[k]['resolution']
    chunk_size = seg_map[k]['chunk_size']
    cv = prepare_precomputed(precomputed_path, offset_xyz, size_xyz, resolution, chunk_size)
    cv[bbox] = seg

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with seg-*/**/*.npz')
  parser.add_argument('output', type=str, default=None,
    help='output_directory')
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  parser.add_argument('--relabel', type=bool, default=True)
  parser.add_argument('--post_clean_up', type=bool, default=False)
  args = parser.parse_args()
  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]

  config_path = os.path.join(args.output, 'stage_0/config.pkl')

  # step 1: MPI read and get local_max
  if mpi_rank == 0:
    seg_list = glob.glob(os.path.join(args.input, 'seg-*'))
    # seg_list.sort()
    os.makedirs(args.output, exist_ok=True)
    sub_indices = np.array_split(np.arange(len(seg_list)), mpi_size)
  else:
    sub_indices = None

  # stage_0_output = os.path.join(args.output, 'stage_0')
  sub_indices = mpi_comm.scatter(sub_indices, 0)
  # seg_map = get_seg_map(args.input, stage_0_output, resolution, chunk_size, sub_indices, args.post_clean_up)
  seg_map = get_seg_map(args.input, args.output, resolution, chunk_size, sub_indices, args.post_clean_up)

  mergeOp = MPI.Op.Create(merge_dict, commute=True)
  seg_map = mpi_comm.allreduce(seg_map, op=mergeOp)

  # step 2: Update local ids with global information and write to cloudvolume 
  mpi_comm.barrier()
  if mpi_rank == 0:
    unify_ids(seg_map)
    #pprint(seg_map)
  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
  update_ids_and_write(seg_map, sub_indices)

  if mpi_rank == 0:
    # stage_out_dir = os.path.join(args.output, 'stage_0')
    # os.makedirs(stage_out_dir, exist_ok=True)
    # with open(os.path.join(stage_out_dir, 'config.pkl'), 'wb') as fp:
    with open(os.path.join(args.output, 'config.pkl'), 'wb') as fp:
      pickle.dump(seg_map, fp)


if __name__ == '__main__':
  main()
