''' Reconciliate postprocess.
For each block, 
1. Remove dust
2. Find large objects covered by mask.
Assimilate their neighbors and prevent them from being 
reconciled/agglomerated.
'''
import numpy as np
import cloudvolume
from cloudvolume.lib import Bbox
import neuroglancer
from matplotlib import rcParams
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import logging
from pprint import pprint
import os
import itertools

from scipy import ndimage
import scipy
import skimage
from skimage.feature import peak_local_max, corner_peaks
from skimage.morphology import local_maxima
from skimage.segmentation import random_walker, watershed
from ffn.utils.bounding_box import OrderlyOverlappingCalculator, BoundingBox
import logging
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm_notebook
import re
import glob

import edt
import fastremap
from skimage.transform import rescale
from pprint import pprint
from collections import defaultdict
from itertools import combinations
from happyneuron.io.utils import rand_cmap
from matplotlib import rcParams
import pickle
from collections import OrderedDict
import argparse

from ffn.inference.segmentation import clear_dust, make_labels_contiguous, clean_up
from happyneuron.ffn.reconciliation.remap import merge_dict, unify_ids, prepare_precomputed
from happyneuron.ffn.reconciliation.remap_cv import get_seg_map_cv

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def process_chunk(in_cv, mask_cv, mask_scale):
  seg_chunk = np.array(in_cv[...])[..., 0]
  in_bb = in_cv.meta.bounds(0)
  in_bb_scale = Bbox(in_bb.minpt // mask_scale, in_bb.maxpt // mask_scale)
  print(in_bb_scale)
  mask_chunk = np.array(mask_cv[in_bb_scale])[..., 0]

  seg_chunk_scale = seg_chunk[::2, ::2, :]
  masked_segs = np.unique(seg_chunk_scale[np.logical_or(mask_chunk == 1, mask_chunk == 2)])
  core_segs = []
  for ms in masked_segs[1:]:
    obj_mask = seg_chunk_scale == ms
    masked_count = np.sum(np.logical_and(obj_mask, mask_chunk))
    total_count = np.sum(obj_mask)
    masked_ratio = masked_count / total_count
    if masked_count > 10000 and masked_ratio > 0.75:
      core_segs.append((ms, masked_count))
  if not core_segs:
    logging.warning('no masked seg')
    return None
  else:
    core_segs = sorted(core_segs, key=lambda kv: kv[1])[::-1]
    logging.warning('masked core seg: %s', core_segs[0])
    return core_segs[0]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with precom-*')
  parser.add_argument('output', type=str, default=None,
    help='output_directory with precom-*')
  parser.add_argument('mask', type=str, default=None,
    help='mask precomputed')
  parser.add_argument('--mask_scale', type=str, default='2,2,1',
    help='mask scale')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()

  config_path = os.path.join(args.output, 'config.pkl')
  remapped_path = os.path.join(args.output, 'remapped')
  mask_scale = [int(i) for i in args.mask_scale.split(',')]

  # step 1: MPI read and get local_max
  if mpi_rank == 0:
    cv_list = glob.glob(os.path.join(args.input, 'precomputed*'))
    os.makedirs(args.output, exist_ok=True)
    sub_indices = np.array_split(np.arange(len(cv_list)), mpi_size)
  else:
    sub_indices = None

  sub_indices = mpi_comm.scatter(sub_indices, 0)
  print(sub_indices)
  # with open(graph_path, 'rb') as fp:
  #   G = pickle.load(fp)
  # with open(config_path, 'rb') as fp:
  #   config = pickle.load(fp)

  mask_cv = cloudvolume.CloudVolume('file://%s' % args.mask, parallel=False, progress=False)

  for i in sub_indices[:10]:
    print(cv_list[i])
    in_cv = cloudvolume.CloudVolume('file://%s' % cv_list[i], parallel=False, progress=False)
    process_chunk(in_cv, mask_cv, mask_scale)


  # seg_map = get_seg_map_cv(args.input, remapped_path, sub_indices, args.post_clean_up, args.verbose)

  mpi_comm.barrier()
  # mergeOp = MPI.Op.Create(merge_dict, commute=True)
  # seg_map = mpi_comm.reduce(seg_map, op=mergeOp, root=0)

  # # step 2: Update local ids with global information and write to cloudvolume 
  # if mpi_rank == 0:
  #   unify_ids(seg_map)
  # else:
  #   seg_map = None
  # seg_map = mpi_comm.bcast(seg_map, 0)
  # update_ids_and_write_cv(seg_map, sub_indices, args.verbose)

  # if mpi_rank == 0:
  #   with open(config_path, 'wb') as fp:
  #     pickle.dump(seg_map, fp)
 
if __name__ == '__main__':
  main()
 