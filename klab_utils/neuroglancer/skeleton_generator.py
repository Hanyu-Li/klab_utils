from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume.lib import Vec
from queue import Queue
import numpy as np
from tqdm import tqdm
import sys
import json
import argparse
import os
import subprocess

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('labels', default=None,
                      help="path to precomputed labels")
  parser.add_argument('--verbose', '-v', action="store_true",
                      help="wether to use progressbar")
  args = parser.parse_args()

  in_path = 'file://'+args.labels
  mip = 0

  # First Pass: Generate Skeletons
  with LocalTaskQueue(parallel=True) as tq:
    tasks = tc.create_skeletonizing_tasks(
        in_path, mip, 
        shape=(512, 512, 512), 
        # see Kimimaro's documentation for the below parameters
        #   teasar_params={ ... }, # controls skeletonization 
        teasar_params={
            'scale': 4,
            'const': 500, # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100, # physical units
            'soma_acceptance_threshold': 3500, # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300, # physical units
            'max_paths': 15, # default None
        },
        object_ids=None, # object id mask if applicable
        progress=args.verbose
    )
    tq.insert_all(tasks)
    # Second Pass: Fuse Skeletons
    tasks = tc.create_skeleton_merge_tasks(
        in_path, mip, 
        crop=0, # in voxels
        magnitude=3, # same as mesh manifests
        dust_threshold=4000, # in nm
        tick_threshold=6000, # in nm
        delete_fragments=False
    )
    tq.insert_all(tasks)
  skel_info_path = os.path.join(args.labels, 'skeletons_mip_0/info')
  with open(skel_info_path, 'w') as f:
    json.dump(
      {
        "@type": "neuroglancer_skeletons",
        "transform": [1,0,0,0,0,1,0,0,0,0,1,0],
        "vertex_attributes": []
      },
      f)


if __name__ == '__main__':
  main()