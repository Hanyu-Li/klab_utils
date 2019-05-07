"""Run with
mpirun -n 16 mesh_generator --labels $LABEL_PATH
"""
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume.lib import Vec
from queue import Queue
import numpy as np
from tqdm import tqdm
import sys
import argparse
import os
import subprocess

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--labels', default=None,
                      help="path to precomputed labels")
  parser.add_argument('--verbose', default=False,
                      help="wether to use progressbar")
  parser.add_argument('--dim_size', default='64,64,64',
                      help="mesh chunksize")
  # parser.add_argument('--simplification_factor', default=10, type=int,
  #                     help="simplification_factor")
  parser.add_argument('--max_simplification_error', default=40, type=int,
                      help="max_simplification_error")

  args = parser.parse_args()

  in_path = 'file://'+args.labels
  mip = 0
  dim_size = tuple(int(d) for d in args.dim_size.split(','))
  print(dim_size)

  print("Making meshes...")
  with LocalTaskQueue(parallel=True) as tq:
    tasks = tc.create_meshing_tasks(in_path, mip=mip, shape=(256, 256, 256))
    tq.insert_all(tasks)
    tasks = tc.create_mesh_manifest_tasks(in_path)
    tq.insert_all(tasks)
  print("Done!")

  #command = r'gunzip {}/mesh/*.gz'.format(args.labels)
  #print(command)
  #subprocess.call(command, shell=True)
  #print("Done!")


if __name__ == '__main__':
  main()
