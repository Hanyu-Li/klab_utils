'''Fourth Stage in reconciliate of ffn inference sub volumes.

Workflow:
1. Assume object ids in subvols are globally unique as output from first stage.
3. Find overlap dict between pairs of precomputed volumes and generate object id pairs
4. Form global reconciliation graph and remap labels 
5. Write into one final precomputed cube


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
import time
import h5py

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def get_bbox_from_cv(cvol):
  offset = np.array(cvol.info['scales'][0]['voxel_offset'])
  size = np.array(cvol.info['scales'][0]['size'])
  return Bbox(offset, offset + size)
def get_merge_dict(G):
  source_cube = nx.get_node_attributes(G, 'cube')
  conn = nx.connected_components(G)

  unique_cubes = np.unique(list(source_cube.values()))
  global_merge_dict = {uid:{} for uid in unique_cubes}
  pbar = tqdm(conn, desc='Find remap for each component')
  for c in pbar:
    c = list(c)
    c.sort()
    for _c in c:
      src_cube = source_cube[_c]
      global_merge_dict[src_cube][_c] = c[0] 
  return global_merge_dict 




def prepare_precomputed(
  precomputed_path, 
  offset, 
  size, 
  resolution, 
  chunk_size, 
  factor=(2,2,1), 
  dtype='uint32',
  parallel=False
  ):
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=parallel)
  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='segmentation',
    data_type=dtype,
    encoding='compressed_segmentation',
    # encoding='raw',
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
def merge_g_op(a, b, datatype):
  return nx.compose(a, b)

# def get_union_bbox_and_merge_path(seg_map, merge_output):
#   bbox_list = [v['bbox'] for v in seg_map.values()]
#   minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
#   maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
#   union_offset = minpt
#   union_size = maxpt - minpt
#   union_bbox = Bbox(minpt, maxpt)
#   cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (merge_output, 
#                                                          union_offset[0], union_offset[1], union_offset[2],
#                                                          union_size[0],  union_size[1], union_size[2])
#   return union_bbox, cv_merge_path
def prewrite(union_bbox, cv_merge_path, resolution, chunk_size):
  union_offset = np.array(union_bbox.minpt)
  union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
  padded_union_size = ((union_size-1) // chunk_size + 1 ) * chunk_size

  cv_merge = prepare_precomputed(
    cv_merge_path, 
    offset=union_offset, 
    size=padded_union_size, 
    resolution=resolution, 
    chunk_size=chunk_size)
  # Pre paint the cv with 0
  # cv_merge[union_bbox] = np.zeros((union_size), dtype=np.uint32)
  
  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)

  for z_start in range(union_offset[2], union_offset[2] + union_size[2], chunk_size[2]):
    logging.warning('z: %d', z_start)
    # cv_merge[:,:,z_start:z_start+chunk_size[2]] = np.zeros((union_size[0], union_size[1], chunk_size[2]), dtype=np.uint32)
    cv_merge[:,:,z_start:z_start+chunk_size[2]] = np.zeros((padded_union_size[0], padded_union_size[1], chunk_size[2]), dtype=np.uint32)
  logging.info('prewrite_finished')

def get_union_bbox_and_merge_path(seg_map, merge_output, global_offset):
  seg_map = {k: v for k, v in seg_map.items() if v}
  bbox_list = [v['bbox'] for v in seg_map.values()]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt + global_offset
  union_size = maxpt - minpt
  # union_bbox = Bbox(minpt, maxpt)
  union_bbox = Bbox(union_offset, union_offset + union_size)
  cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (merge_output, 
                                                         union_offset[0], union_offset[1], union_offset[2],
                                                         union_size[0],  union_size[1], union_size[2])
  return union_bbox, cv_merge_path
def get_chunk_bboxes(union_bbox, chunk_size):
  ffn_style_bbox = bounding_box.BoundingBox(
    np.array(union_bbox.minpt), np.array(union_bbox.size3()))

  calc = bounding_box.OrderlyOverlappingCalculator(
    outer_box=ffn_style_bbox, 
    sub_box_size=chunk_size, 
    overlap=[0,0,0], 
    include_small_sub_boxes=True,
    back_shift_small_sub_boxes=False)
  bbs = [ffn_bb for ffn_bb in calc.generate_sub_boxes()]
  for ffn_bb in bbs:
    logging.warning('sub_bb: %s', ffn_bb)
  return bbs

def h5_to_cloudvolume(h5_path, cv_path, union_offset, local_sub_bboxes, resolution, 
  chunk_size, global_offset=(0, 0, 0), flip_h5=False):

  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=False, parallel=False)
  cv_merge = cloudvolume.CloudVolume('file://' + cv_path, mip=0, **cv_args)
  with h5py.File(h5_path, 'r') as f:
    h5_ds = f['output']
    pbar = tqdm(local_sub_bboxes, desc='h5 to precomputed')
    for ffn_bb in pbar:
      abs_offset = ffn_bb.start
      abs_size = ffn_bb.size

      rel_offset = abs_offset - union_offset
      rel_size = abs_size

      # logging.warning('write %s %s', abs_offset, abs_size)
      h5_slc = np.s_[
        rel_offset[0]:rel_offset[0] + rel_size[0],
        rel_offset[1]:rel_offset[1] + rel_size[1],
        rel_offset[2]:rel_offset[2] + rel_size[2]
      ]

      cv_slc = np.s_[
        abs_offset[0]:abs_offset[0] + abs_size[0],
        abs_offset[1]:abs_offset[1] + abs_size[1],
        abs_offset[2]:abs_offset[2] + abs_size[2],
        0
      ]

      cv_merge[cv_slc] = h5_ds[h5_slc]
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with remapped/precomputed-*, config.pkl, and graph.pkl')
  parser.add_argument('--output', type=str, default=None,
    help='output_directory')
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  parser.add_argument('--batch_scale', type=int, default=1,
    help='Controls how much data is loaded from h5 each time, by multiplying chunk_size')
  parser.add_argument('--global_offset', type=str, default='0,0,0')
  parser.add_argument('--flip_h5', type=bool, default=False)
  parser.add_argument('--verbose', type=bool, default=True)
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(level='DEBUG')
  else:
    logging.basicConfig(level='ERROR')
  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]
  global_offset = [int(i) for i in args.global_offset.split(',')]

  if args.output is None:
    output = args.input
  config_path = os.path.join(args.input, 'config.pkl')


  if mpi_rank == 0:
    assert os.path.exists(config_path), 'Run reconciliate_remap first'
    with open(config_path, 'rb') as fp:
      seg_map = pickle.load(fp)
    os.makedirs(output, exist_ok=True)
  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
    
  merge_output = os.path.join(output, 'agglomerated')
  h5_path = os.path.join(output, 'intermediate.h5')

  if mpi_rank == 0:
    union_bbox, cv_merge_path = get_union_bbox_and_merge_path(seg_map, merge_output, global_offset)

    # preset precomputed
    union_offset = np.array(union_bbox.minpt)
    union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
    cv_merge = prepare_precomputed(
      cv_merge_path, 
      offset=union_offset, 
      size=union_size, 
      resolution=resolution, 
      chunk_size=chunk_size)
    
    # sub divide aligned bboxes
    sub_bbox_size = [i * args.batch_scale for i in chunk_size]

    bbs = get_chunk_bboxes(union_bbox, sub_bbox_size)
    sub_bbs = np.array_split(bbs, mpi_size) 
    logging.warn('write shapes %s %s', union_bbox, sub_bbox_size)

  else:
    # union_bbox = None
    union_offset = None
    cv_merge_path = None
    sub_bbs = None


  union_offset = mpi_comm.bcast(union_offset, 0)
  cv_merge_path = mpi_comm.bcast(cv_merge_path, 0)
  sub_bbs = mpi_comm.scatter(sub_bbs, 0)

  h5_to_cloudvolume(h5_path, cv_merge_path, union_offset, sub_bbs, 
    resolution, chunk_size, args.flip_h5)
  sys.exit()


if __name__ == '__main__':
  main()
