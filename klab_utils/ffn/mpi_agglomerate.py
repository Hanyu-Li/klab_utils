"""Agglomerate two neighboring/overlapping cloud-volumes"""
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
#     print(bb)
    data = data[::scaling[0], ::scaling[1], ::scaling[2]]
    cvol = cloudvolume.CloudVolume('file://'+cv_path, mip=m, **cv_args)
    cvol[bb] = data
#     print(data.shape)
    pass
def agglomerate(cv_path_1, cv_path_2, contiguous=False, inplace=False, 
                no_zero=True):
  """Given two cloudvolumes and bounding boxes, intersect and perform agglomeration"""
  
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=True)
  
  cv1 = cloudvolume.CloudVolume('file://'+cv_path_1, mip=0, **cv_args)
  cv2 = cloudvolume.CloudVolume('file://'+cv_path_2, mip=0, **cv_args)

#   bb1 = get_bbox(cv_path_1)
#   bb2 = get_bbox(cv_path_2)
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
    # filter out all with either key or val == 0
    remap_label = {k:v for k, v in remap_label.items() if k != 0 and v != 0}
  data_2_full = cv2[bb2]
  data_2_full_remap = perform_remap(data_2_full, remap_label)
  if inplace:
    cv2[bb2] = data_2_full_remap
    update_mips(cv_path_2, bb2, **cv_args)
  
  
  return remap_label
  

def prepare_precomputed(precomputed_path, offset, size, resolution, chunk_size, dtype='uint32'):
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
#     volume_size=size_xyz,
    volume_size=np.array(size),
    chunk_size=chunk_size,
    max_mip=0,
    factor=(2,2,1),
    )
  cv = CloudVolume('file://'+precomputed_path, mip=0, info=info, **cv_args)
  cv.commit_info()
  return cv
def write_to_precomputed(seg, bbox, precomputed_path, resolution, chunk_size):
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=True)
  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='segmentation',
    data_type=seg.dtype,
    encoding='compressed_segmentation',
    resolution=list(resolution),
    voxel_offset=np.array(bbox.minpt),
#     volume_size=size_xyz,
    volume_size=np.array(bbox.maxpt) - np.array(bbox.minpt),
    chunk_size=chunk_size,
    max_mip=0,
    factor=(2,2,1),
    )
  cv = CloudVolume('file://'+precomputed_path, mip=0, info=info, **cv_args)
  cv.commit_info()
  cv[bbox] = seg
def agglomerate_segs(input_dir, 
                     output_dir,
                     resolution=None,
                     chunk_size=None):
    
  get_name = lambda s: re.sub('seg-', 'precomputed-', s)
  seg_list = glob.glob(os.path.join(input_dir, 'seg*'))
  # seg_list = glob.glob(os.path.join(input_dir, 'seg*'))
  seg_list.sort()
  print(seg_list)
  
  seg_map = dict()
  global_max = 0
  for i, s in tqdm(enumerate(seg_list)):
    seg, offset_zyx = load_inference(s)
    offset_xyz = offset_zyx[::-1]
    size_xyz = seg.shape[::-1]

    bbox = Bbox(a=offset_xyz, b=offset_xyz+size_xyz)
    seg = np.transpose(seg, [2, 1, 0])
    
    # make labels contiguous and unique globally but keep 0 intact
    seg = np.uint32(seg)
    zeros = seg==0
    seg, _ = make_labels_contiguous(seg)
    seg = np.uint32(seg)
    seg += global_max
    global_max = np.max(seg)
    seg[zeros] = 0

    # convert to precomputed
    
    precomputed_path = os.path.join(output_dir, get_name(os.path.basename(s)))
    seg_map[i] = (bbox, precomputed_path)
    cv = prepare_precomputed(precomputed_path, offset_xyz, size_xyz, resolution, chunk_size)
    cv[bbox] = seg

  G = nx.Graph()
  G.add_nodes_from(seg_map.keys())
  # find neighbors
  for k1, k2 in itertools.product(seg_map.keys(), seg_map.keys()):
    bb1, p1 = seg_map[k1]
    bb2, p2 = seg_map[k2]
    int_bb = Bbox.intersection(bb1, bb2)
    if not int_bb.empty() and k1 != k2:
      G.add_edge(k1, k2)

  # agglomerate each pair and rewrite cloud volume
  for k1, k2 in G.edges():
    _, p1 = seg_map[k1]
    _, p2 = seg_map[k2]
    remap_label  = agglomerate(p1, p2, contiguous=False, inplace=True, no_zero=True)
    print(len(remap_label))
  bbox_list = [v for k,(v,_) in seg_map.items()]
  print(bbox_list)
  return seg_map

def merge(seg_map, resolution, chunk_size):
  bbox_list = [v for (v,_) in seg_map.values()]
  minpt = np.min(np.stack([np.array(b.minpt) for b in bbox_list], 0), axis=0)
  maxpt = np.max(np.stack([np.array(b.maxpt) for b in bbox_list], 0), axis=0)
  
  union_offset = minpt
  union_size = maxpt - minpt
  union_bbox = Bbox(minpt, maxpt)
  print(union_bbox)
  # create new canvas
  cv_merge_path = '%s/precomputed-%d_%d_%d_%d_%d_%d/' % (os.path.dirname(seg_map[0][1]), 
                                                         union_offset[0], union_offset[1], union_offset[2],
                                                         union_size[0],  union_size[1], union_size[2])
  print(cv_merge_path)
  cv_merge = prepare_precomputed(cv_merge_path, offset=union_offset, size=union_size, resolution=resolution, 
                      chunk_size=chunk_size)
  print(cv_merge.shape)
  cv_merge[union_bbox] = np.zeros((union_size), dtype=np.uint32)
  
  cv_args = dict(
    bounded=True, fill_missing=False, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
  
  cvs = []
  val_dict = dict()
  for i, (bb, precom) in tqdm(seg_map.items()):
#     print(bb, precom)
    cv = CloudVolume('file://'+precom, mip=0, **cv_args)
#     val = cv.download_to_shared_memory(np.s_[:], str(i))
    val = cv[...]
    val_dict[bb] = val
    cv_merge[bb] = val
                                                         
def get_seg_map(input_dir, output_dir):
  get_name = lambda s: re.sub('seg-', 'precomputed-', s)
  seg_list = glob.glob(os.path.join(input_dir, 'seg*'))
  seg_list.sort()
  
  seg_map = dict()
  # global_max = 0
  for i, s in tqdm(enumerate(seg_list)):
    # get_source = lambda s: glob.glob(os.path.join(s, '**/*.npz', recursive=True)[0]
    seg, offset_zyx = load_inference(s)
    # seg_name = get_source(s)
    # offset_zyx = get_zyx(seg_name)
    offset_xyz = offset_zyx[::-1]
    size_xyz = seg.shape[::-1]

    bbox = Bbox(a=offset_xyz, b=offset_xyz+size_xyz)
    # seg = np.transpose(seg, [2, 1, 0])
    
    # # make labels contiguous and unique globally but keep 0 intact
    # seg = np.uint32(seg)
    # zeros = seg==0
    # seg, _ = make_labels_contiguous(seg)
    # seg = np.uint32(seg)
    # seg += global_max
    # global_max = np.max(seg)
    # seg[zeros] = 0

    # convert to precomputed
    
    precomputed_path = os.path.join(output_dir, get_name(os.path.basename(s)))
    # seg_map[i] = (bbox, precomputed_path)
    # cv = prepare_precomputed(precomputed_path, offset_xyz, size_xyz, resolution, chunk_size)
    # cv[bbox] = seg
    seg_map[i] = (bbox, s, precomputed_path)
  return seg_map
def infer_grid_shape():
  pass
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with seg-*/**/*.npz')
  parser.add_argument('output', type=str, default=None,
    help='output_directory')
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')
  args = parser.parse_args()
  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]

  # step 1: MPI convert all to cloudvolume
  if mpi_rank == 0:
    seg_map = get_seg_map(args.input, args.output)
    pprint(seg_map)
    

  # seg_map = agglomerate_segs(args.input, args.output, resolution=resolution, 
  #   chunk_size=chunk_size)
  # merge(seg_map, resolution, chunk_size)


  
