'''Graph Reduce approach 

1. Traverse each sub volume and ensure global unique segments

2. Find overlap dict between pairs of sub volumes

3. Reduce 

'''
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
# from cloudvolume import CloudVolume
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


  mergeOp = MPI.Op.Create(merge_dict, commute=True)
  seg_map = mpi_comm.allreduce(seg_map, op=mergeOp)


if __name__ == '__main__':
  main()