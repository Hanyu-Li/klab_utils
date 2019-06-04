"""Agglomerate neighboring/overlapping cloud-volumes"""
import cloudvolume
import re
import numpy as np
from pprint import pprint
import argparse
import fastremap
from cloudvolume.lib import Bbox
from ffn.inference.segmentation import make_labels_contiguous

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
from ffn.utils.bounding_box import intersection
import itertools
import networkx as nx
import json
import pickle

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def get_bbox(path):
  ox, oy, oz, sx, sy, sz = [int(i) for i in re.search(r'seg-(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', path).groups()]
  offset = np.asarray([ox, oy, oz])
  size = np.asarray([sx, sy, sz])
  return Bbox(offset, offset + size)


def get_bbox_from_cv(cvol):
  offset = np.array(cvol.info['scales'][0]['voxel_offset'])
  size = np.array(cvol.info['scales'][0]['size'])
  return Bbox(offset, offset + size)


def find_remap(a, b):
  """find relabel map from a to b"""
  flat_a = a.ravel()
  flat_b = b.ravel()
  flat_ab = np.stack((flat_a,flat_b), axis=1)
  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      flat_ab, return_inverse=True, return_counts=True, axis=0)
  max_overlap_ids = dict()
  for (label_a, label_b), count in zip(unique_joint_labels, joint_counts):
    new_pair = (label_b, count)
    existing = max_overlap_ids.setdefault(label_a, new_pair)
    if existing[1] < count:
      max_overlap_ids[label_a] = new_pair
  relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  return relabel_map


def perform_remap(a, relabel_map):
  remapped_a = fastremap.remap(a, relabel_map, preserve_missing_labels=True) 
  return remapped_a


def update_mips(cv_path, bb, **kwargs):
  cvol = cloudvolume.CloudVolume('file://'+cv_path, mip=0, **kwargs)
  last_res = cvol.info['scales'][0]['resolution']
  data = cvol[bb]
  for m, s in enumerate(cvol.info['scales']):
    if m == 0:
      continue
    curr_res = np.array(s['resolution'])
    scaling = curr_res // last_res
    last_res = curr_res
    bb = bb // scaling
    data = data[::scaling[0], ::scaling[1], ::scaling[2]]
    cvol = cloudvolume.CloudVolume('file://'+cv_path, mip=m, **cv_args)
    cvol[bb] = data


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
  if no_zero:
    # filter out ones with either key or val == 0
    remap_label = {k:v for k, v in remap_label.items() if k != 0 and v != 0}
  data_2_full = cv2[bb2]
  data_2_full_remap = perform_remap(data_2_full, remap_label)
  if inplace:
    cv2[bb2] = data_2_full_remap
    update_mips(cv_path_2, bb2, **cv_args)
  
  return remap_label
  
def agglomerate_v2(cv_path_1, cv_path_2, contiguous=False, inplace=False, 
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
  if no_zero:
    # filter out ones with either key or val == 0
    remap_label = {k:v for k, v in remap_label.items() if k != 0 and v != 0}
  data_2_full = cv2[bb2]
  data_2_full_remap = perform_remap(data_2_full, remap_label)
  if inplace:
    cv2[bb2] = data_2_full_remap
    update_mips(cv_path_2, bb2, **cv_args)
  
  return remap_label

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
    encoding='compressed_segmentation',
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

                                                         
def get_seg_map(input_dir, output_dir, resolution, chunk_size, sub_indices):
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
    seg, offset_zyx = load_inference(seg_path)
    offset_xyz = offset_zyx[::-1]
    size_xyz = seg.shape[::-1]

    bbox = Bbox(a=offset_xyz, b=offset_xyz+size_xyz)
    seg = np.transpose(seg, [2, 1, 0])
    
    # make labels contiguous and unique globally but keep 0 intact
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
      'input': seg_path, 
      'output': precomputed_path, 
      'resolution': resolution,
      'chunk_size': chunk_size,
      'local_max': local_max}
  return seg_map

def unify_ids(seg_map):
  keys = list(seg_map.keys())
  keys.sort()
  #pprint(keys)
  global_max = 0
  for k in keys:
    seg_map[k]['global_offset'] = global_max
    global_max += seg_map[k]['local_max']


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


def infer_grid_v0(seg_map):
  corners = []
  for k,v in seg_map.items():
    bb = v['bbox']
    x, y, z = bb.minpt
    corners.append((x, y, z))
  pass
  corners = np.stack(corners, 0)
  corners = corners - corners[0]
  gcd = np.gcd.reduce(corners)
  grid = corners // gcd
  #print(grid)
  #print('>>')
  #pprint(seg_map)
  keys = list(seg_map.keys())
  keys.sort()
  for i, k in enumerate(keys):
    seg_map[k]['grid']=grid[i]
  
  # each set is grouped into one 8-set serial agglomeration
  max_grid = len(seg_map.keys())
  if max_grid > 8: 
    grid_set = np.array([
      [0,0,0],
      [0,0,1],
      [0,1,0],
      [0,1,1],
      [1,0,0],
      [1,0,1],
      [1,1,0],
      [1,1,1],
    ])
  else:
    grid_set = grid

  # divid into sets
  grid_shape = grid[-1]+1
  group_id = 0
  # grid_to_id_map = {tuple(grid[i]):i for i in range(len(grid))}
  grid_to_id_map = {tuple(grid[i]):k for i,k in enumerate(keys)}
  for offset in itertools.product(range(0, grid_shape[0], 2),
                            range(0, grid_shape[1], 2),
                            range(0, grid_shape[2], 2)):
    group_ids = grid_set + np.array(offset)
    for i in group_ids:
      if tuple(i) in grid_to_id_map:
        j = grid_to_id_map[tuple(i)] 
        seg_map[j]['group_id'] = group_id
    group_id += 1
  return group_id
def infer_grid(seg_map):
  corners = []
  corner_str = []
  corner_keys = []
  for k,v in seg_map.items():
    bb = v['bbox']
    x, y, z = bb.minpt
    corners.append((x, y, z))
    corner_str.append('%s_%s_%s' % (str(x).zfill(6), str(y).zfill(6), str(z).zfill(6)))
    corner_keys.append(k)
  
  order = np.argsort(corner_str, axis=0)
  #print('order:', order)
  
  corners = np.stack(corners, 0)
  corners = corners[order,:]
  corner_keys = np.array(corner_keys)[order]
  
  min_corner = np.min(corners, 0)
  corners = corners - min_corner
  logging.warning('corners: %s', str(corners))
  gcd = np.gcd.reduce(corners)
  gcd[gcd==0] = 1
  # print(gcd)
  grid = corners // gcd
  logging.warning('grid: %s', str(grid))
  for i, k in enumerate(corner_keys):
    seg_map[k]['grid']=grid[i]

  # each set is grouped into one 8-set serial agglomeration
  max_grid = len(seg_map.keys())
  if max_grid > 8:
    grid_set = np.array([
      [0,0,0],
      [0,0,1],
      [0,1,0],
      [0,1,1],
      [1,0,0],
      [1,0,1],
      [1,1,0],
      [1,1,1],
    ])
  else:
    grid_set = grid

  # divid into sets
  grid_shape = grid[-1]+1
  #print(grid_shape)
  group_id = 0
  grid_to_id_map = {tuple(grid[i]):k for i,k in enumerate(corner_keys)}
  for offset in itertools.product(range(0, grid_shape[0], 2),
                            range(0, grid_shape[1], 2),
                            range(0, grid_shape[2], 2)):
    group_ids = grid_set + np.array(offset)
    for i in group_ids:
      if tuple(i) in grid_to_id_map:
        j = grid_to_id_map[tuple(i)]
        seg_map[j]['group_id'] = group_id
    group_id += 1
  return group_id

def agglomerate_group(seg_map, merge_output, gid=None, relabel=True):
  if seg_map == {}:
    return {}
  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())

  # find neighbors
  for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
    bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
    bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
    int_bb = Bbox.intersection(bb1, bb2)
    if not int_bb.empty() and k1 != k2:
      G.add_edge(k1, k2)

  # agglomerate each pair and rewrite cloud volume
  for k1, k2 in G.edges():
    p1 = seg_map[k1]['output']
    p2 = seg_map[k2]['output']
    if relabel:
      remap_label  = agglomerate(p1, p2, contiguous=False, inplace=True, no_zero=True)
  return merge(seg_map, merge_output, gid)

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

def agglomerate_group_v2(seg_map, merge_output, gid=None, relabel=True):
  if seg_map == {}:
    return {}
  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())

  # find neighbors
  for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
    bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
    bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
    int_bb = Bbox.intersection(bb1, bb2)
    if not int_bb.empty() and k1 != k2:
      G.add_edge(k1, k2)

  G_overlap = nx.Graph()
  # agglomerate each pair and rewrite cloud volume
  pbar = tqdm(G.edges(), total=G.number_of_edges(), desc='find overlap')
  global_merge_dict = {}
  for k1, k2 in pbar:
    p1 = seg_map[k1]['output']
    p2 = seg_map[k2]['output']
    #if relabel:
    remap_label  = agglomerate_v2(p1, p2, contiguous=False, inplace=False, no_zero=True)
    # a dict of object ids from p2 -> p1
    #src_attr = {k:{'cube': k2} for k in remap_label.keys()}
    #tgt_attr = {k:{'cube': k1} for k in remap_label.keys()}


    G_overlap.add_nodes_from(remap_label.keys(), cube=k2)
    G_overlap.add_nodes_from(remap_label.values(), cube=k1)
    edges = [(k, v) for k, v in remap_label.items()]
    G_overlap.add_edges_from(edges)
    global_merge_dict = get_merge_dict(G_overlap)


    
  # pprint('rank %d, %s' % (mpi_rank, global_merge_dict))
  # with open('test_graph%d.pkl' % mpi_rank, 'wb') as fp:
  #   pickle.dump(G_overlap, fp)

  return merge_graph(seg_map, merge_output, global_merge_dict, gid)

def stage_wise_agglomerate(seg_map, output, stage=0):
  pass
  if mpi_rank == 0:
    max_group_id = infer_grid(seg_map)
    #print('max_group_id', max_group_id)
    group_subset = np.array_split(np.arange(max_group_id), mpi_size)
  else:
    seg_map = None
    group_subset = None
  seg_map = mpi_comm.bcast(seg_map, 0)
  group_subset = mpi_comm.scatter(group_subset)
  #print('>> group_subset', group_subset)

  # step 4: perform (stage wise) merge operation 
  grouped_seg_map = {k:{} for k in group_subset}
  for k, v in seg_map.items():
    if 'group_id' not in v:
      print('error>>', v)
    gid = v['group_id']
    if gid in group_subset:
      grouped_seg_map[gid][k] = v 

  #logging.warning('>> keys %d, %d', mpi_rank, len(grouped_seg_map.keys()))

  prev_stage = os.path.join(output, 'stage_%d' % stage)

  # recursively merge
  stage = stage + 1
  merge_output = os.path.join(output, 'stage_%d' % stage)
  stage_seg_map = {}
  for gid in group_subset:
    stage_seg_map = agglomerate_group(grouped_seg_map[gid], merge_output, gid)
    stage += 1
  mergeOp = MPI.Op.Create(merge_dict, commute=True)
  stage_seg_map = mpi_comm.allreduce(stage_seg_map, op=mergeOp)
  # if mpi_rank == 0:
  #   print('>>>>', stage_seg_map)
  return stage_seg_map

def merge(seg_map, merge_output, gid=None):
  resolution = list(seg_map.values())[0]['resolution']
  chunk_size = list(seg_map.values())[0]['chunk_size']
  bbox_list = [v['bbox'] for v in seg_map.values()]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt
  union_size = maxpt - minpt
  union_bbox = Bbox(minpt, maxpt)
  #print(union_bbox)
  # create new canvas
  cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (merge_output, 
                                                         union_offset[0], union_offset[1], union_offset[2],
                                                         union_size[0],  union_size[1], union_size[2])
  #print(cv_merge_path)
  cv_merge = prepare_precomputed(cv_merge_path, offset=union_offset, size=union_size, resolution=resolution, 
                      chunk_size=chunk_size)
  #print(cv_merge.shape)
  # Pre paint the cv with 0
  cv_merge[union_bbox] = np.zeros((union_size), dtype=np.uint32)
  
  cv_args = dict(
    bounded=True, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
  
  #val_dict = dict()
  #print('>>>>rank: %d, map_keys %s' % (mpi_rank, str(seg_map.keys())))

  pbar = tqdm(seg_map.values(), desc='merging')
  for seg in pbar:
    bb = seg['bbox']
    cv = CloudVolume('file://'+seg['output'], mip=0, **cv_args)
#     val = cv.download_to_shared_memory(np.s_[:], str(i))
    val = cv[...]
    #logging.error('rank %d val_shape: %s, bbox %s', mpi_rank, val.shape, bb)
    #val_dict[bb] = val
    curr_val = cv_merge[bb][:]
    non_zeros = curr_val != 0
    val[non_zeros] = curr_val[non_zeros]
    cv_merge[bb] = val
  return {gid: dict(
    bbox=union_bbox,
    output=cv_merge_path,
    resolution=resolution,
    chunk_size=chunk_size,
  )}
def merge_graph(seg_map, merge_output, global_merge_dict, gid=None):
  resolution = list(seg_map.values())[0]['resolution']
  chunk_size = list(seg_map.values())[0]['chunk_size']
  bbox_list = [v['bbox'] for v in seg_map.values()]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt
  union_size = maxpt - minpt
  union_bbox = Bbox(minpt, maxpt)
  #print(union_bbox)
  # create new canvas
  cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (merge_output, 
                                                         union_offset[0], union_offset[1], union_offset[2],
                                                         union_size[0],  union_size[1], union_size[2])
  #print(cv_merge_path)
  cv_merge = prepare_precomputed(cv_merge_path, offset=union_offset, size=union_size, resolution=resolution, 
                      chunk_size=chunk_size)
  #print(cv_merge.shape)
  # Pre paint the cv with 0
  cv_merge[union_bbox] = np.zeros((union_size), dtype=np.uint32)
  
  cv_args = dict(
    bounded=True, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
  
  #val_dict = dict()
  #print('>>>>rank: %d, map_keys %s' % (mpi_rank, str(seg_map.keys())))

  pbar = tqdm(seg_map.items(), desc='merging')
  for seg_key, seg in pbar:
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
    val[non_zeros] = curr_val[non_zeros]
    cv_merge[bb] = val
  return {gid: dict(
    bbox=union_bbox,
    output=cv_merge_path,
    resolution=resolution,
    chunk_size=chunk_size,
  )}
def stage_wise_agglomerate_v2(seg_map, output, group_subset, stage=0, relabel=True):

  # step 4: perform (stage wise) merge operation 
  grouped_seg_map = {k:{} for k in group_subset}
  for k, v in seg_map.items():
    if 'group_id' not in v:
      print('error>>', v)
    gid = v['group_id']
    if gid in group_subset:
      grouped_seg_map[gid][k] = v 

  #print('>>', mpi_rank, len(grouped_seg_map.keys()))
  #print('rank: %d >>>>>group_seg_map %s' %(mpi_rank,  grouped_seg_map))


  # recursively merge
  # stage = stage + 1
  merge_output = os.path.join(output, 'stage_%d' % stage)
  stage_seg_map = {}
  for gid in group_subset:
    # stage_seg_map[gid] = agglomerate_group(grouped_seg_map[gid], merge_output, gid, relabel)
    ret_dict = agglomerate_group(grouped_seg_map[gid], merge_output, gid, relabel)
    assert gid in ret_dict
    stage_seg_map[gid] = ret_dict[gid]
    #stage += 1
  # if mpi_rank == 0:
  #   print('>>>>', stage_seg_map)
  return stage_seg_map


def stage_wise_agglomerate_v3(seg_map, output, group_subset, stage=0, relabel=True):

  # step 4: perform (stage wise) merge operation 
  grouped_seg_map = {k:{} for k in group_subset}
  for k, v in seg_map.items():
    if 'group_id' not in v:
      print('error>>', v)
    gid = v['group_id']
    if gid in group_subset:
      grouped_seg_map[gid][k] = v 

  #print('>>', mpi_rank, len(grouped_seg_map.keys()))
  #print('rank: %d >>>>>group_seg_map %s' %(mpi_rank,  grouped_seg_map))


  # recursively merge
  # stage = stage + 1
  merge_output = os.path.join(output, 'stage_%d' % stage)
  stage_seg_map = {}
  for gid in group_subset:
    # stage_seg_map[gid] = agglomerate_group(grouped_seg_map[gid], merge_output, gid, relabel)
    ret_dict = agglomerate_group_v2(grouped_seg_map[gid], merge_output, gid, relabel)
    assert gid in ret_dict
    stage_seg_map[gid] = ret_dict[gid]
    #stage += 1
  # if mpi_rank == 0:
  #   print('>>>>', stage_seg_map)
  return stage_seg_map

def sequential_agglomerate(seg_map, merge_output):
  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())

  # find neighbors
  for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
    bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
    bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
    int_bb = Bbox.intersection(bb1, bb2)
    if not int_bb.empty() and k1 != k2:
      G.add_edge(k1, k2)

  # agglomerate each pair and rewrite cloud volume
  for k1, k2 in G.edges():
    p1 = seg_map[k1]['output']
    p2 = seg_map[k2]['output']
    remap_label = agglomerate(p1, p2, contiguous=False, inplace=True, no_zero=True)
    #print(len(remap_label))
  return merge(seg_map, merge_output)

def merge_dict(a, b, datatype):
  return {**a, **b}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with seg-*/**/*.npz')
  parser.add_argument('output', type=str, default=None,
    help='output_directory')
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  parser.add_argument('--relabel', type=bool, default=True)
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
    seg_map = get_seg_map(args.input, stage_0_output, resolution, chunk_size, sub_indices)

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
  ####
  # sequential agglomeration 
  #merge_output = os.path.join(args.output, 'stage_final')
  #if mpi_rank == 0:
  #  sequential_agglomerate(seg_map, merge_output)

  # Step 3: Agglomerate recursively
  relabel = args.relabel
  stage = 1
  while len(seg_map.keys()) > 1:
    mpi_comm.barrier()
    if mpi_rank == 0:
      max_group_id = infer_grid(seg_map)
      group_subset = np.array_split(np.arange(max_group_id), mpi_size)
    else:
      seg_map = None
      group_subset = None
    seg_map = mpi_comm.bcast(seg_map, 0)
    group_subset = mpi_comm.scatter(group_subset)
    seg_map = stage_wise_agglomerate_v3(seg_map, args.output, group_subset, stage=stage, relabel=relabel) 

    mpi_comm.barrier()
    mergeOp = MPI.Op.Create(merge_dict, commute=True)
    seg_map = mpi_comm.allreduce(seg_map, op=mergeOp)
    if mpi_rank == 0:
      stage_out_dir = os.path.join(args.output, 'stage_%d' % stage)
      os.makedirs(stage_out_dir, exist_ok=True)
      with open(os.path.join(stage_out_dir, 'config.pkl'), 'wb') as fp:
        pickle.dump(seg_map, fp)
    stage += 1



  
