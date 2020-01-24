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
  relabel_map = {k:v[0] for k,v in max_overlap_ids.items()}
  return relabel_map
  # return max_overlap_ids

def perform_remap(a, relabel_map):
  remapped_a = fastremap.remap(a, relabel_map, preserve_missing_labels=True) 
  return remapped_a

  
def agglomerate(cv_path_1, cv_path_2, overlap_thresh, contiguous=False, inplace=False, 
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
  # find remap from 2 to 1, {label_2: (label_1, overlap_counts)}
  remap_label = find_remap(data_2, data_1, overlap_thresh)
  # if no_zero:
  #   # filter out ones with either key or val == 0
  #   remap_label = {k:v for k, v in remap_label.items() if k != 0 and v != 0}
  # data_2_full = cv2[bb2]
  # data_2_full_remap = perform_remap(data_2_full, remap_label)
  if inplace:
    data_2_full = cv2[bb2]
    data_2_full_remap = perform_remap(data_2_full, remap_label)
    cv2[bb2] = data_2_full_remap
    update_mips(cv_path_2, bb2, **cv_args)
  
  return remap_label
def get_neighbor_graph(seg_map):
  start = time.time()
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
  end = time.time()
  logging.warning('Get neighbor graph took: %s seconds', end-start)
  
  return G

def agglomerate_group_graph(seg_map, G_neighbor, overlap_thresh, edge_indices=None, verbose=True):
  # agglomerate each edge and generate global remap dict 
  full_edge_arr = np.array(G_neighbor.edges())
  local_edge_set = full_edge_arr[edge_indices]
  if verbose:
    pbar = tqdm(local_edge_set, total=len(edge_indices), desc='find overlap')
  else:
    pbar = local_edge_set
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
    remap_label = agglomerate(p1, p2, overlap_thresh, contiguous=False, inplace=False, no_zero=True)
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

def merge_g_op(a, b, datatype):
  return nx.compose(a, b)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with remapped/precomputed-*, and config.pkl')
  # parser.add_argument('output', type=str, default=None,
  #   help='output_directory')
  parser.add_argument('--overlap_thresh', type=int, default=1000)
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  parser.add_argument('--verbose', type=bool, default=False)
  args = parser.parse_args()

  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]

  config_path = os.path.join(args.input, 'config.pkl')
  assert os.path.exists(config_path)
  if mpi_rank == 0:
    with open(config_path, 'rb') as fp:
      seg_map = pickle.load(fp)
    # os.makedirs(args.output, exist_ok=True)
  else:
    seg_map = None
  seg_map = mpi_comm.bcast(seg_map, 0)
    
  # Graph merge:
  graph_path = os.path.join(args.input, 'graph.pkl')
  # merge_output = os.path.join(args.itput

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
  logging.warning('rank %d: %s', mpi_rank, sub_edge_indices)
  G_overlap, local_merge_dict = agglomerate_group_graph(
    seg_map, G_neighbor, args.overlap_thresh, edge_indices=sub_edge_indices, args.verbose)
  logging.warning('rank %d: %s', mpi_rank, (len(G_overlap.edges())))

  mpi_comm.barrier()
  mergeGraphOp = MPI.Op.Create(merge_g_op, commute=True)
  G_overlap = mpi_comm.allreduce(G_overlap, op=mergeGraphOp)
  if mpi_rank == 0:
    # stage_out_dir = os.path.join(args.output, 'stage_0')
    # os.makedirs(stage_out_dir, exist_ok=True)
    with open(graph_path, 'wb') as fp:
      pickle.dump(G_overlap, fp)



if __name__ == '__main__':
  main()
