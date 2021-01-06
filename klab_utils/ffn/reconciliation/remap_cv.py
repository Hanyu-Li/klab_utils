'''Convert processed precomputed subvols to be globally unique.
'''
import cloudvolume
import re
import time
import numpy as np
from pprint import pprint
import argparse
import fastremap
from cloudvolume.lib import Bbox
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

from happyneuron.ffn.reconciliation.remap import merge_dict, unify_ids, prepare_precomputed

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def get_seg_map_cv(input_dir, output_dir, sub_indices, post_clean_up=False, verbose=False):
  if not sub_indices.size:
    return {}
  seg_list = glob.glob(os.path.join(input_dir, 'precomputed*'))
  seg_list.sort()
  
  seg_map = dict()
  if verbose:
    pbar = tqdm(sub_indices, desc='Generating seg map')
  else:
    pbar = sub_indices
  for i in pbar:
    seg_path = seg_list[i]
    seg_cv = cloudvolume.CloudVolume('file://%s' % seg_path, mip=0, 
      progress=False, parallel=False)
    bbox = seg_cv.meta.bounds(0)
    seg = np.array(seg_cv[bbox][..., 0])
    resolution = seg_cv.meta.resolution(0)
    chunk_size = seg_cv.meta.chunk_size(0)
    
    # make labels contiguous and unique globally but keep 0 intact
    min_particle = 1000
    if post_clean_up:
      clean_up(seg, min_particle) 
     
    seg = np.uint32(seg)
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    local_max = np.max(seg)
    
    precomputed_path = os.path.join(
      output_dir, os.path.basename(seg_path))

    seg_map[i] = {
      'bbox': bbox, 
      'input': os.path.abspath(seg_path), 
      'output': os.path.abspath(precomputed_path), 
      'resolution': resolution,
      'chunk_size': chunk_size,
      'local_max': local_max}
  return seg_map


def update_ids_and_write_cv(seg_map, sub_indices, verbose=False):
  if verbose:
    pbar = tqdm(sub_indices, desc='Update ids to be globally unique')
  else:
    pbar = sub_indices
  for k in pbar:
    if not seg_map[k]: continue
    seg_path = seg_map[k]['input']
    # seg, offset_zyx = load_inference(s)
    seg_cv = cloudvolume.CloudVolume('file://%s' % seg_path, mip=0, 
      progress=False, parallel=False)
    bbox = seg_cv.meta.bounds(0)
    seg = np.array(seg_cv[bbox][..., 0])
    # offset_xyz = offset_zyx[::-1]
    # size_xyz = seg.shape[::-1]

    # bbox = seg_map[k]['bbox']
    # seg = np.transpose(seg, [2, 1, 0])
    # # make labels contiguous and unique globally but keep 0 intact
    seg = np.uint32(seg)
    zeros = seg == 0
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    # local_max = np.max(seg)
    seg += seg_map[k]['global_offset']
    seg[zeros] = 0

    # convert to precomputed
    precomputed_path = seg_map[k]['output']
    resolution = seg_map[k]['resolution']
    chunk_size = seg_map[k]['chunk_size']

    out_info = seg_cv.info.copy()
    # out_cv = cloudvo
    # cv = prepare_precomputed(precomputed_path, offset_xyz, size_xyz, resolution, chunk_size)
    # cv[bbox] = seg

    out_cv = cloudvolume.CloudVolume('file://%s' % precomputed_path, info=out_info, mip=0)
    out_cv.commit_info()
    out_cv[...] = seg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with precom-*')
  parser.add_argument('output', type=str, default=None,
    help='output_directory')
  parser.add_argument('--relabel', type=bool, default=True)
  parser.add_argument('--post_clean_up', action='store_true')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  config_path = os.path.join(args.output, 'config.pkl')
  remapped_path = os.path.join(args.output, 'remapped')

  # step 1: MPI read and get local_max
  if mpi_rank == 0:
    cv_list = glob.glob(os.path.join(args.input, 'precomputed*'))
    os.makedirs(args.output, exist_ok=True)
    sub_indices = np.array_split(np.arange(len(cv_list)), mpi_size)
  else:
    sub_indices = None

  sub_indices = mpi_comm.scatter(sub_indices, 0)
  seg_map = get_seg_map_cv(args.input, remapped_path, sub_indices, args.post_clean_up, args.verbose)

  mpi_comm.barrier()
  mergeOp = MPI.Op.Create(merge_dict, commute=True)
  seg_map = mpi_comm.reduce(seg_map, op=mergeOp, root=0)

  # step 2: Update local ids with global information and write to cloudvolume 
  if mpi_rank == 0:
    unify_ids(seg_map)
  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
  update_ids_and_write_cv(seg_map, sub_indices, args.verbose)

  if mpi_rank == 0:
    with open(config_path, 'wb') as fp:
      pickle.dump(seg_map, fp)


if __name__ == '__main__':
  main()
