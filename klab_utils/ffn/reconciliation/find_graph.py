'''Second Stage in reconciliate of ffn inference sub volumes.

Workflow:
1. Assume object ids in subvols are globally unique as output from first stage.
2. Find overlap dict between pairs of precomputed volumes and generate object id pairs
3. Form global reconciliation graph 
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

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def get_bbox_from_cv(cvol):
  offset = np.array(cvol.info['scales'][0]['voxel_offset'])
  size = np.array(cvol.info['scales'][0]['size'])
  return Bbox(offset, offset + size)
# def get_merge_dict(G):
#   source_cube = nx.get_node_attributes(G, 'cube')
#   conn = nx.connected_components(G)

#   unique_cubes = np.unique(list(source_cube.values()))
#   global_merge_dict = {uid:{} for uid in unique_cubes}
#   for c in conn:
#     c = list(c)
#     c.sort()
#     for _c in c:
#       src_cube = source_cube[_c]
#       global_merge_dict[src_cube][_c] = c[0] 
#   return global_merge_dict 


def find_remap(a, b, overlap_thresh=100):
  """find relabel map from a to b"""

  flat_a = a.ravel()
  flat_b = b.ravel()
  flat_ab = np.stack((flat_a,flat_b), axis=1)

  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      flat_ab, return_inverse=True, return_counts=True, axis=0)
  max_overlap_ids = dict()
  for (label_a, label_b), count in zip(unique_joint_labels, joint_counts):
    if label_a == 0 or label_b == 0 or count < overlap_thresh:
      continue
    new_pair = (label_b, count)
    
    existing = max_overlap_ids.setdefault(label_a, new_pair)
    if existing[1] < count:
      max_overlap_ids[label_a] = new_pair
  # relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  relabel_map = {k:v for k,v in max_overlap_ids.items()}
  return relabel_map
  # return max_overlap_ids

def find_remap_v2(a, b, overlap_thresh=100):
  """find relabel map from a to b

  Ensure mutually maximum overlap
  """

  flat_a = a.ravel()
  flat_b = b.ravel()
  flat_ab = np.stack((flat_a,flat_b), axis=1)

  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      flat_ab, return_inverse=True, return_counts=True, axis=0)
  max_overlap_ids_a = dict()
  max_overlap_ids_b = dict()
  relabel_map = dict()
  for (label_a, label_b), count in zip(unique_joint_labels, joint_counts):
    if label_a == 0 or label_b == 0 or count < overlap_thresh:
      continue
    # new_pair = (label_b, count)
    
    existing_a = max_overlap_ids_a.setdefault(label_a, (label_b, count))
    existing_b = max_overlap_ids_b.setdefault(label_b, (label_a, count))
    if existing_a[1] < count:
      max_overlap_ids_a[label_a] = (label_b, count)
    if existing_b[1] < count:
      max_overlap_ids_b[label_b] = (label_a, count)

    if (max_overlap_ids_b[label_b][0] == label_a and 
        max_overlap_ids_a[label_a][0] == label_b):
      relabel_map[label_a] = label_b
      relabel_map[label_a] = (label_b, count)

  # relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  # relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  # relabel_map = {k:v for k,v in mutual_max_pairs.items()}
  return relabel_map

def perform_remap(a, relabel_map):
  remapped_a = fastremap.remap(a, relabel_map, preserve_missing_labels=True) 
  return remapped_a

  
def agglomerate(cv_path_1, cv_path_2, overlap_thresh, contiguous=False, 
                mutual_exclusive=False,
                no_zero=True):
  """Given two cloudvolumes, intersect and perform agglomeration"""
  cv_args = dict(
    bounded=True, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
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
  if mutual_exclusive:
    remap_label = find_remap_v2(data_2, data_1, overlap_thresh)
  else:
    remap_label = find_remap(data_2, data_1, overlap_thresh)

  
  return remap_label

def get_neighbor_graph(seg_map):
  start = time.time()
  if seg_map == {}:
    return {}
  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())
  valid_keys = [k for k in seg_map.keys() if seg_map[k]]

  # find neighbors
  # for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
  for k1, k2 in itertools.product(valid_keys, valid_keys):
    bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
    bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
    if Bbox.intersects(bb1, bb2) and k1 != k2:
      G.add_edge(k1, k2)
  end = time.time()
  logging.warning('Get neighbor graph took: %s seconds', end-start)
  
  return G

def get_neighbor_graph_par(seg_map, sub_keys):
  start = time.time()
  if seg_map == {}:
    return {}
  G = nx.Graph()
  sorted_keys = sorted(seg_map.keys())
  G.add_nodes_from(sub_keys)

  # find neighbors
  pbar = tqdm(sub_keys, desc='find neighbors')
  for k1 in pbar:
    for k2 in sorted_keys:
      if k2 <= k1:
        continue
      if not seg_map[k1] or not seg_map[k2]:
        continue
      # logging.warning('rank %d\t pair %d %d' % (mpi_rank, k1, k2))
      bb1, p1 = seg_map[k1]['bbox'], seg_map[k1]['output']
      bb2, p2 = seg_map[k2]['bbox'], seg_map[k2]['output']
      if Bbox.intersects(bb1, bb2) and k1 != k2:
        G.add_edge(k1, k2)
  end = time.time()
  logging.warning('Get neighbor graph took: %s seconds', end-start)
  return G

def get_object_sizes(seg_path):
  seg_cv = cloudvolume.CloudVolume('file://%s' % seg_path, progress=False, parallel=False)
  seg_chunk = np.array(seg_cv[...])[..., 0]
  uni, counts = np.unique(seg_chunk, return_counts=True)
  nz = uni != 0
  uni = uni[nz]
  counts = counts[nz]
  return {k:v for k, v in zip(uni, counts)}

def agglomerate_group_graph(seg_map, G_neighbor, overlap_thresh, edge_indices=None, 
  mutual_exclusive=False, verbose=True):
  # agglomerate each edge and generate global remap dict 
  full_edge_arr = np.array(list(G_neighbor.edges()))
  # print(G_neighbor.edges())
  local_edge_set = full_edge_arr[edge_indices]
  pbar = tqdm(local_edge_set, total=len(edge_indices), disable=not verbose, desc='find overlap')
  global_merge_dict = {}
  G_overlaps = []
  size_maps = {}
  for k1, k2 in pbar:
    G_overlap = nx.Graph()
    p1 = seg_map[k1]['output']
    p2 = seg_map[k2]['output']
    remap_label = agglomerate(
      p1, p2, overlap_thresh, contiguous=False, 
      mutual_exclusive=mutual_exclusive,
      no_zero=True)
    
    if k1 in size_maps:
      size_map_1 = size_maps[k1]
    else:
      size_map_1 = get_object_sizes(p1)
      size_maps[k1] = size_map_1
    if k2 in size_maps:
      size_map_2 = size_maps[k2]
    else:
      size_map_2 = get_object_sizes(p2)
      size_maps[k2] = size_map_2

    # G_overlap.add_nodes_from(remap_label.keys(), cube=k2)
    # G_overlap.add_nodes_from([v[0] for v in remap_label.values()], cube=k1)

    node_map_2 = {k: {'size': size_map_2[k], 'cube': k2} for k in remap_label.keys()}
    node_map_1 = {v[0]: {'size': size_map_1[v[0]], 'cube': k1} for v in remap_label.values()}

    # node_map_1 = {k: {'size': size_map_1[k], 'cube': k1} for k in size_map_1.keys()}
    # node_map_2 = {k: {'size': size_map_2[k], 'cube': k2} for k in size_map_2.keys()}

    G_overlap.add_nodes_from(node_map_2.items())
    G_overlap.add_nodes_from(node_map_1.items())


    edges = [(k, v[0]) for k, v in remap_label.items()]
    weights = {(k, v[0]):v[1] for k, v in remap_label.items()}
    G_overlap.add_edges_from(edges)
    nx.set_edge_attributes(G_overlap, weights, 'weight')
    # edge_weights = [v[1] for k, v in remap_label.items()]
    G_overlaps.append(G_overlap)
  if len(G_overlaps):
    G_overlaps_merged = nx.compose_all(G_overlaps)
  else:
    G_overlaps_merged = nx.Graph()
  # global_merge_dict = get_merge_dict(G_overlaps_merged)
  return G_overlaps_merged #, global_merge_dict

def merge_g_op(a, b, datatype):
  return nx.compose(a, b)

def update_seg_map(seg_map, segmentation_dir):
  '''Force 'output' entry to using segmentation dir.'''
  updated_seg_map = {}
  for k, v in seg_map.items():
    if 'output' in v:
      in_f = v['output']
      bname = os.path.basename(in_f)
      out_f = os.path.join(segmentation_dir, bname)
      v['output'] = out_f
    updated_seg_map[k] = v
  return updated_seg_map

def parse_disconnect_pairs(disconnect_pairs_txt):
  pairs = []
  with open(disconnect_pairs_txt, 'r') as f:
    for line in f.readlines():
      l, r = line.split(',')
      pairs.append((int(l), int(r)))
  return pairs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with remapped/precomputed-*, and config.pkl')
  # parser.add_argument('output', type=str, default=None,
  #   help='output_directory')
  parser.add_argument('--segmentation_dir', type=str, default=None, 
    help='Dir of precomputed-* when input is different from original config')
  parser.add_argument('--config_path', type=str, default=None)
  parser.add_argument('--out_config_path', type=str, default=None)
  parser.add_argument('--graph_path', type=str, default=None)
  parser.add_argument('--disconnect_pairs', type=str, default=None)
  parser.add_argument('--overlap_thresh', type=int, default=500)
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  parser.add_argument('--mutual_exclusive', action='store_true')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]

  if mpi_rank == 0:
    if not args.config_path:
      config_path = os.path.join(args.input, 'config.pkl')
    else:
      config_path = args.config_path
    assert os.path.exists(config_path)

    with open(config_path, 'rb') as fp:
      seg_map = pickle.load(fp)
    if args.segmentation_dir:
      assert args.out_config_path is not None
      seg_map = update_seg_map(seg_map, os.path.abspath(args.segmentation_dir))
      with open(args.out_config_path, 'wb') as fp:
        pickle.dump(seg_map, fp)

  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
    
  # Graph merge:
  if not args.graph_path:
    graph_path = os.path.join(args.input, 'graph.pkl')
  else:
    graph_path = args.graph_path
  mpi_comm.barrier()

  mergeGraphOp = MPI.Op.Create(merge_g_op, commute=True)
  if mpi_rank == 0:
    keys = list(seg_map.keys())
    keys = np.array(sorted(keys))
    sub_keys = np.array_split(keys, mpi_size)
    # print(sub_keys)
  else:
    sub_keys = None
  sub_keys = mpi_comm.scatter(sub_keys, 0)

  print('rank %d, keys: %s' % (mpi_rank, sub_keys))
  G_neighbor = get_neighbor_graph_par(seg_map, sub_keys)
  G_neighbor = mpi_comm.allreduce(G_neighbor, op=mergeGraphOp)
  mpi_comm.barrier()

  if mpi_rank == 0:
    sub_edge_indices = np.array_split(np.arange(len(G_neighbor.edges())), mpi_size)
  else:
    sub_edge_indices = None
  sub_edge_indices = mpi_comm.scatter(sub_edge_indices, 0)



  G_overlap = agglomerate_group_graph(
    seg_map, G_neighbor, args.overlap_thresh, edge_indices=sub_edge_indices, 
    mutual_exclusive=args.mutual_exclusive,
    verbose=args.verbose)

  logging.warning('rank %d reached barrier', mpi_rank)
  mpi_comm.barrier()
  logging.warning('rank %d passed barrier', mpi_rank)
  # G_overlap = mpi_comm.allreduce(G_overlap, op=mergeGraphOp)
  G_overlap = mpi_comm.reduce(G_overlap, op=mergeGraphOp, root=0)
  if mpi_rank == 0:
    # stage_out_dir = os.path.join(args.output, 'stage_0')
    # os.makedirs(stage_out_dir, exist_ok=True)

    # post correction, remove according to a txt file of id pairs
    if args.disconnect_pairs:
      disconnect_pairs = parse_disconnect_pairs(args.disconnect_pairs)
      pprint(disconnect_pairs)
      G_overlap.remove_edges_from(disconnect_pairs)

    with open(graph_path, 'wb') as fp:
      pickle.dump(G_overlap, fp)



if __name__ == '__main__':
  main()
