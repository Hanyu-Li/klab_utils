'''Third Stage in reconciliate of ffn inference sub volumes.

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



def perform_remap(a, relabel_map):
  remapped_a = fastremap.remap(a, relabel_map, preserve_missing_labels=True) 
  return remapped_a

def prepare_precomputed(precomputed_path, offset, size, resolution, chunk_size, factor=(2,2,1), dtype='uint32'):
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
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
def merge_g_op(a, b, datatype):
  return nx.compose(a, b)

def get_union_bbox_and_merge_path(seg_map, merge_output):
  bbox_list = [v['bbox'] for v in seg_map.values() if v]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt
  union_size = maxpt - minpt
  union_bbox = Bbox(minpt, maxpt)
  cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (merge_output, 
                                                         union_offset[0], union_offset[1], union_offset[2],
                                                         union_size[0],  union_size[1], union_size[2])
  return union_bbox, cv_merge_path
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

def remap_and_write(seg_map, union_bbox, cv_merge_path, global_merge_dict, sub_indices):
  seg_map = {k: v for k, v in seg_map.items() if v}
  resolution = list(seg_map.values())[0]['resolution']
  chunk_size = list(seg_map.values())[0]['chunk_size']
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
    progress=False, provenance=None, compress=False, 
    non_aligned_writes=True, parallel=False)
  
  #val_dict = dict()
  #print('>>>>rank: %d, map_keys %s' % (mpi_rank, str(seg_map.keys())))

  # pbar = tqdm(seg_map.items(), desc='merging')
  pbar = tqdm(sub_indices, desc='merging')
  # for seg_key, seg in pbar:
  for seg_key in pbar:
    seg = seg_map[seg_key]
    bb = seg['bbox']
    cv = CloudVolume('file://'+seg['output'], mip=0, **cv_args)
#     val = cv.download_to_shared_memory(np.s_[:], str(i))
    val = cv[...]
    # print('keys: %s <-> %s' % (seg_key, global_merge_dict.keys()))
    if seg_key in global_merge_dict:
      val = perform_remap(val, global_merge_dict[seg_key])
    #logging.error('rank %d val_shape: %s, bbox %s', mpi_rank, val.shape, bb)
    #val_dict[bb] = val

    curr_val = cv_merge[bb][:]
    non_zeros = curr_val != 0
    logging.warn('loaded curr val %s', np.sum(non_zeros[:]))
    val[non_zeros] = curr_val[non_zeros]
    try:
      cv_merge[bb] = val
      logging.warning('finished writing')
    except:
      logging.warning('write racing')
      time.sleep(5)
      cv_merge[bb] = val
      logging.warning('rewritten')


  return dict(
    bbox=union_bbox,
    output=cv_merge_path,
    resolution=resolution,
    chunk_size=chunk_size,
  )
def get_union_bbox_and_merge_path(seg_map, merge_output):
  bbox_list = [v['bbox'] for v in seg_map.values() if v]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt
  union_size = maxpt - minpt
  union_bbox = Bbox(minpt, maxpt)
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
def prewrite_h5(union_bbox, h5_path, resolution, chunk_size, flip_h5):
  union_offset = np.array(union_bbox.minpt)
  union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)

  if flip_h5: 
    union_size = union_size[::-1]
  with h5py.File(h5_path, 'w') as f:
    logging.warning('created %s, %s', union_size, chunk_size)
    # f.create_dataset('output', shape=tuple(union_size), dtype=np.uint32, chunks=tuple(chunk_size), fillvalue=0)
    # f.create_dataset('output', shape=tuple(union_size), dtype=np.uint32, chunks=tuple(chunk_size))
    f.create_dataset('output', shape=tuple(union_size), dtype=np.uint32)
def remap_and_write_h5(seg_map, union_bbox, h5_path, 
  global_merge_dict, sub_indices, flip_h5):
  seg_map = {k: v for k, v in seg_map.items() if v}
  resolution = list(seg_map.values())[0]['resolution']
  chunk_size = list(seg_map.values())[0]['chunk_size']
  union_offset = np.array(union_bbox.minpt)
  union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
  padded_union_size = ((union_size-1) // chunk_size + 1 ) * chunk_size
  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=False, 
    non_aligned_writes=False, parallel=False)
  if flip_h5: 
    union_size = union_size[::-1]

  with h5py.File(h5_path, 'w', driver='mpio', comm=mpi_comm) as f:
    # logging.warning(f.keys())
    # ds = f['output']
    ds = f.create_dataset('output', shape=tuple(union_size), dtype=np.uint32)
    # chunks=(16, 16, 16))
    # logging.warn('rank: %d sees %s', mpi_rank, ds.shape)
    pbar = tqdm(sub_indices, desc='merging')
    for seg_key in pbar:
      seg = seg_map[seg_key]
      bb = seg['bbox']
      cv = CloudVolume('file://'+seg['output'], mip=0, **cv_args)
  #     val = cv.download_to_shared_memory(np.s_[:], str(i))
      val = np.array(cv[...])[...,0]

      # logging.warning('from cv %s', val.shape)
      # relative slc in h5 write
      slc = np.s_[
        bb.minpt[0]-union_offset[0]:bb.maxpt[0]-union_offset[0],
        bb.minpt[1]-union_offset[1]:bb.maxpt[1]-union_offset[1],
        bb.minpt[2]-union_offset[2]:bb.maxpt[2]-union_offset[2]
      ]
      if flip_h5:
        slc = slc[::-1]
        val = np.transpose(val, (2,1,0))

      if seg_key in global_merge_dict:
        val = perform_remap(val, global_merge_dict[seg_key])
      curr_val = ds[slc][:]
      # logging.warning('from h5 %s', curr_val.shape)
      non_zeros = curr_val != 0
      # logging.warning('non_zeros %s', np.sum(non_zeros[:]))
      val[non_zeros] = curr_val[non_zeros]

      with ds.collective:
        ds[slc] = val[...]

def h5_to_cloudvolume(h5_path, cv_path, union_bbox, sub_bboxes, sub_indices, resolution, chunk_size):
  union_offset = np.array(union_bbox.minpt)
  union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
  cv_merge = prepare_precomputed(
    cv_path, 
    offset=union_offset, 
    size=union_size, 
    resolution=resolution, 
    chunk_size=chunk_size)

  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=False, parallel=False)
  local_sub_bboxes = [sub_bboxes[i] for i in sub_indices]
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
  parser.add_argument('--flip_h5', type=bool, default=False)
  args = parser.parse_args()
  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]

  if args.output is None:
    output = args.input
  else:
    output = args.output
  config_path = os.path.join(args.input, 'config.pkl')
  graph_path = os.path.join(args.input, 'graph.pkl')


  if mpi_rank == 0:
    assert os.path.exists(config_path), 'Run reconciliate_remap first'
    assert os.path.exists(graph_path), 'Run reconciliate_find_graph first'
    with open(config_path, 'rb') as fp:
      seg_map = pickle.load(fp)
    with open(graph_path, 'rb') as fp:
      G_overlap = pickle.load(fp)
    os.makedirs(output, exist_ok=True)
  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
    
  merge_output = os.path.join(output, 'agglomerated')

  # remove h5 dependency
  # if os.path.exists(h5_path):
  #   # already writen h5 path, skip this step
  #   logging.warning('Found h5 file, skipped h5 writing')
  # else:

  # (off) perform parallel write to an mpi h5 file to avoid racing

  # h5 mode
  h5_path = os.path.join(output, 'intermediate.h5')
  if mpi_rank == 0:
    global_remap_dict = get_merge_dict(G_overlap)
    sub_indices = np.array_split(list(global_remap_dict.keys()), mpi_size)
    # set up h5 dataset canvas
    union_bbox, cv_merge_path = get_union_bbox_and_merge_path(seg_map, merge_output)
    # prewrite(union_bbox, cv_merge_path, resolution, chunk_size)
    # prewrite_h5(union_bbox, h5_path, resolution, chunk_size, args.flip_h5)

  else:
    global_remap_dict = None
    sub_indices = None
    union_bbox = None
    cv_merge_path = None
    h5_path = None

  global_remap_dict = mpi_comm.bcast(global_remap_dict, 0)
  sub_indices = mpi_comm.scatter(sub_indices, 0)
  union_bbox = mpi_comm.bcast(union_bbox, 0)
  cv_merge_path = mpi_comm.bcast(cv_merge_path, 0)
  h5_path = mpi_comm.bcast(h5_path, 0)

  logging.warning('diagnose: %s %s', union_bbox, h5_path)
  remap_and_write_h5(seg_map, union_bbox, h5_path, global_remap_dict, sub_indices, args.flip_h5)

  mpi_comm.barrier()




if __name__ == '__main__':
  main()
