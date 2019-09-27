'''Reconciliate subvolumes of segmentation from ffn inferences

Workflow:
1. Traverse each sub volume and ensure object ids are globally unique
2. Convert each .npz file to precomputed format
3. Find overlap dict between pairs of precomputed volumes and generate object id pairs
4. Form global reconciliation graph and remap labels 
5. Write into one final precomputed cube

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
from klab_utils.ffn.export_inference import load_inference, get_zyx
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


def get_bbox_from_cv(cvol):
  offset = np.array(cvol.info['scales'][0]['voxel_offset'])
  size = np.array(cvol.info['scales'][0]['size'])
  return Bbox(offset, offset + size)
def get_merge_dict(G):
  source_cube = nx.get_node_attributes(G, 'cube')
  conn = nx.connected_components(G)

  unique_cubes = np.unique(list(source_cube.values()))
  global_merge_dict = {uid:{} for uid in unique_cubes}
  for c in conn:
    c = list(c)
    c.sort()
    for _c in c:
      src_cube = source_cube[_c]
      global_merge_dict[src_cube][_c] = c[0] 
  return global_merge_dict 
# def find_remap(a, b):
#   """find relabel map from a to b"""
#   flat_a = a.ravel()
#   flat_b = b.ravel()
#   flat_ab = np.stack((flat_a,flat_b), axis=1)
#   unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
#       flat_ab, return_inverse=True, return_counts=True, axis=0)
#   max_overlap_ids = dict()
#   for (label_a, label_b), count in zip(unique_joint_labels, joint_counts):
#     new_pair = (label_b, count)
#     existing = max_overlap_ids.setdefault(label_a, new_pair)
#     if existing[1] < count:
#       max_overlap_ids[label_a] = new_pair
#   relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
#   return relabel_map
def find_remap(a, b, thresh=100):
  """find relabel map from a to b"""
  flat_a = a.ravel()
  flat_b = b.ravel()
  flat_ab = np.stack((flat_a,flat_b), axis=1)
  
  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      flat_ab, return_inverse=True, return_counts=True, axis=0)
  
  max_overlap_ids = dict()
  for (label_a, label_b), count in zip(unique_joint_labels, joint_counts):
    if label_a == 0 or label_b == 0 or count < thresh:
      continue
    new_pair = (label_b, count)
    
    existing = max_overlap_ids.setdefault(label_a, new_pair)
    if existing[1] < count:
      max_overlap_ids[label_a] = new_pair
  relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  return relabel_map

def perform_remap(a, relabel_map):
  remapped_a = fastremap.remap(a, relabel_map, preserve_missing_labels=True) 
  return remapped_a


def get_seg_map(input_dir, output_dir, resolution, chunk_size, sub_indices, post_clean_up=False):
  if not sub_indices.size:
    return {}
  get_name = lambda s: re.sub('seg-', 'precomputed-', s)
  seg_list = glob.glob(os.path.join(input_dir, 'seg*'))
  seg_list.sort()
  
  seg_map = dict()
  pbar = tqdm(sub_indices, desc='Generating seg map')
  for i in pbar:
  # for i in tqdm(sub_indices):
    # pbar.set_description("Generating Seg Map:")
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
      # non_background = (seg > 0).astype(np.uint8)
      # seg = connected_components(non_background)

    seg = np.uint32(seg)
    #zeros = seg==0
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    local_max = np.max(seg)
    #seg += global_max
    # global_max = np.max(seg)
    # seg[zeros] = 0

    
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
  
def agglomerate(cv_path_1, cv_path_2, contiguous=False, inplace=False, 
                no_zero=True):
  """Given two cloudvolumes, intersect and perform agglomeration"""
  cv_args = dict(
    bounded=True, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=True)
  
  cv1 = cloudvolume.CloudVolume('file://'+cv_path_1, mip=0, **cv_args)
  cv2 = cloudvolume.CloudVolume('file://'+cv_path_2, mip=0, **cv_args)

  bb1 = get_bbox_from_cv(cv1)
  bb2 = get_bbox_from_cv(cv2)
  
  int_bb = Bbox.intersection(bb1, bb2)
  data_1 = cv1[int_bb]
  data_2 = cv2[int_bb]
  if contiguous:
    data_1, map_1 = make_labels_contiguous(data_1)
    data_2, map_2 = make_labels_contiguous(data_2)
    
  data_1 = np.uint32(data_1)
  data_2 = np.uint32(data_2)
  # find remap from 2 to 1
  remap_label = find_remap(data_2, data_1)
  # if no_zero:
  #   # filter out ones with either key or val == 0
  #   remap_label = {k:v for k, v in remap_label.items() if k != 0 and v != 0}
  data_2_full = cv2[bb2]
  data_2_full_remap = perform_remap(data_2_full, remap_label)
  if inplace:
    cv2[bb2] = data_2_full_remap
    update_mips(cv_path_2, bb2, **cv_args)
  
  return remap_label
def get_neighbor_graph(seg_map):
  if seg_map == {}:
    return {}
  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())

  # find neighbors
  for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
    bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
    bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
    if Bbox.intersects(bb1, bb2) and k1 != k2:
      G.add_edge(k1, k2)
  return G

def agglomerate_group_graph(seg_map, G_neighbor, merge_output, edge_indices=None):
  # agglomerate each edge and generate global remap dict 
  full_edge_arr = np.array(G_neighbor.edges())
  local_edge_set = full_edge_arr[edge_indices]
  pbar = tqdm(local_edge_set, total=len(edge_indices), desc='find overlap')
  # pbar = tqdm(G_neighbor.edges(), total=G_neighbor.number_of_edges(), desc='find overlap')
  global_merge_dict = {}
  # counter = 0
  G_overlaps = []
  for k1, k2 in pbar:
    G_overlap = nx.Graph()
    # logging.warning('test %s %s', k1, k2)
    p1 = seg_map[k1]['output']
    p2 = seg_map[k2]['output']
    #if relabel:
    remap_label = agglomerate(p1, p2, contiguous=False, inplace=False, no_zero=True)
    # a dict of object ids from p2 -> p1
    #src_attr = {k:{'cube': k2} for k in remap_label.keys()}
    #tgt_attr = {k:{'cube': k1} for k in remap_label.keys()}


    G_overlap.add_nodes_from(remap_label.keys(), cube=k2)
    G_overlap.add_nodes_from(remap_label.values(), cube=k1)
    edges = [(k, v) for k, v in remap_label.items()]
    G_overlap.add_edges_from(edges)
    G_overlaps.append(G_overlap)
    # counter += 1
    # if counter == 3:
    #   break
  if len(G_overlaps):
    G_overlaps_merged = nx.compose_all(G_overlaps)
  else:
    G_overlaps_merged = nx.Graph()
  global_merge_dict = get_merge_dict(G_overlaps_merged)
  return G_overlaps_merged, global_merge_dict


def unify_ids(seg_map):
  keys = list(seg_map.keys())
  keys.sort()
  #pprint(keys)
  global_max = 0
  for k in keys:
    seg_map[k]['global_offset'] = global_max
    global_max += seg_map[k]['local_max']
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
def merge_dict(a, b, datatype):
  return {**a, **b}
def merge_g_op(a, b, datatype):
  return nx.compose(a, b)

def get_union_bbox_and_merge_path(seg_map, merge_output):
  bbox_list = [v['bbox'] for v in seg_map.values()]
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
      logging.info('finished writing')
    except:
      logging.warn('write racing')
      time.sleep(5)
      cv_merge[bb] = val
      logging.warn('rewritten')


  return dict(
    bbox=union_bbox,
    output=cv_merge_path,
    resolution=resolution,
    chunk_size=chunk_size,
  )

def get_chunk_bboxes(union_bbox, chunk_size):
  # union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
  # padded_union_size = ((union_size-1) // chunk_size + 1 ) * chunk_size
  ffn_style_bbox = bounding_box.BoundingBox(
    np.array(union_bbox.minpt), np.array(union_bbox.size3()))
  # ffn_style_bbox = bounding_box.BoundingBox(
  #   np.array(union_bbox.minpt), padded_union_size)

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
  if os.path.exists(config_path):
    if mpi_rank == 0:
      with open(config_path, 'rb') as fp:
        seg_map = pickle.load(fp)
    else:
      seg_map = None
    seg_map = mpi_comm.bcast(seg_map, 0)
    
  else:
    # step 1: MPI read and get local_max
    if mpi_rank == 0:
      seg_list = glob.glob(os.path.join(args.input, 'seg-*'))
      seg_list.sort()
      sub_indices = np.array_split(np.arange(len(seg_list)), mpi_size)
    else:
      sub_indices = None

    stage_0_output = os.path.join(args.output, 'stage_0')
    sub_indices = mpi_comm.scatter(sub_indices, 0)
    seg_map = get_seg_map(args.input, stage_0_output, resolution, chunk_size, sub_indices, args.post_clean_up)

    mergeOp = MPI.Op.Create(merge_dict, commute=True)
    seg_map = mpi_comm.allreduce(seg_map, op=mergeOp)

    # step 2: Update local ids with global information and write to cloudvolume 
    if mpi_rank == 0:
      unify_ids(seg_map)
      #pprint(seg_map)
    else:
      seg_map = None
    seg_map = mpi_comm.bcast(seg_map, 0)
    update_ids_and_write(seg_map, sub_indices)

    if mpi_rank == 0:
      stage_out_dir = os.path.join(args.output, 'stage_0')
      os.makedirs(stage_out_dir, exist_ok=True)
      with open(os.path.join(stage_out_dir, 'config.pkl'), 'wb') as fp:
        pickle.dump(seg_map, fp)
  # Graph merge:
  # graph_path = os.path.join(args.output, 'stage_1_h5/graph.pkl')
  graph_path = os.path.join(args.output, 'stage_1/graph.pkl')
  os.makedirs(os.path.dirname(graph_path), exist_ok=True)

  # h5_path = os.path.abspath(os.path.join(args.output, 'stage_1_h5/test.h5'))
  merge_output = os.path.join(args.output, 'stage_1')
  # union_bbox, cv_merge_path = get_union_bbox_and_merge_path(seg_map, merge_output)

  if os.path.exists(graph_path):
    # already writen h5 path, skip this step
    logging.warning('Found graph, skipped graph gen')
    with open(graph_path, 'rb') as fp:
      G_overlap = pickle.load(fp)
  else:
    mpi_comm.barrier()
    if mpi_rank == 0:
      G_neighbor = get_neighbor_graph(seg_map)
      # print(G_neighbor.edges())
      sub_edge_indices = np.array_split(np.arange(len(G_neighbor.edges())), mpi_size)
    else:
      G_neighbor = None
      sub_edge_indices = None
    G_neighbor = mpi_comm.bcast(G_neighbor, 0)
    sub_edge_indices = mpi_comm.scatter(sub_edge_indices, 0)
      # pprint(seg_map)
    logging.warning('rank %d: %s', mpi_rank, sub_edge_indices)
    G_overlap, local_merge_dict = agglomerate_group_graph(seg_map, G_neighbor, merge_output, edge_indices=sub_edge_indices)
    logging.warning('rank %d: %s', mpi_rank, (len(G_overlap.edges())))

    mpi_comm.barrier()
    mergeGraphOp = MPI.Op.Create(merge_g_op, commute=True)
    G_overlap = mpi_comm.allreduce(G_overlap, op=mergeGraphOp)
    if mpi_rank == 0:
      # stage_out_dir = os.path.join(args.output, 'stage_0')
      # os.makedirs(stage_out_dir, exist_ok=True)
      with open(graph_path, 'wb') as fp:
        pickle.dump(G_overlap, fp)

  # remove h5 dependency
  # if os.path.exists(h5_path):
  #   # already writen h5 path, skip this step
  #   logging.warning('Found h5 file, skipped h5 writing')
  # else:

  # perform parallel write to an mpi h5 file to avoid racing
  if mpi_rank == 0:
    global_remap_dict = get_merge_dict(G_overlap)
    sub_indices = np.array_split(list(global_remap_dict.keys()), mpi_size)
    union_bbox, cv_merge_path = get_union_bbox_and_merge_path(seg_map, merge_output)
    prewrite(union_bbox, cv_merge_path, resolution, chunk_size)

  else:
    global_remap_dict = None
    sub_indices = None
    union_bbox = None
    cv_merge_path = None

  global_remap_dict = mpi_comm.bcast(global_remap_dict, 0)
  sub_indices = mpi_comm.scatter(sub_indices, 0)
  union_bbox = mpi_comm.bcast(union_bbox, 0)
  cv_merge_path = mpi_comm.bcast(cv_merge_path, 0)

  remap_and_write(seg_map, union_bbox, cv_merge_path, global_remap_dict, sub_indices)


  # final stage: write from h5 to a cloud volume 
  mpi_comm.barrier()

  sys.exit()
  
  



if __name__ == '__main__':
  main()
