"""Run with
mpirun -n 16 mesh_generator --labels $LABEL_PATH
"""
from taskqueue import TaskQueue, MockTaskQueue
import igneous.task_creation as tc
from cloudvolume.lib import Vec
from queue import Queue
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import logging
import sys
import json
import argparse
import os
import re
from collections import defaultdict
import subprocess
from igneous.tasks import MeshManifestTask
from .mesh_generator import mpiTaskQueue
from taskqueue import RegisteredTask, LocalTaskQueue
from cloudvolume.storage import Storage

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()


def build_segid_map(layer_path):
  with Storage(layer_path) as storage:
    _info = json.loads(storage.get_file('info').decode('utf8'))
    mesh_dir = os.path.join(layer_path.replace('file://', ''), _info['mesh'])
    # if mesh_dir is None and 'mesh' in self._info:
      # self.mesh_dir = self._info['mesh']
    print(mesh_dir)
    segids = defaultdict(list)

    # pbar = tqmd()
    for entry in tqdm(os.scandir(mesh_dir), desc='Build Segid map'):
      match = re.match(r'(\d+):(\d+).*.gz', entry.name)
      if not match: 
        continue
      fname, segid, lod = match.group(0), int(match.group(1)), int(match.group(2))
      segids[segid].append(fname)

    
    
    return segids, _info['mesh']


def parallel_build_segid_map(layer_path):
  with Storage(layer_path) as storage:
    _info = json.loads(storage.get_file('info').decode('utf8'))
    mesh_dir = os.path.join(layer_path.replace('file://', ''), _info['mesh'])
    # if mesh_dir is None and 'mesh' in self._info:
      # self.mesh_dir = self._info['mesh']
    print(mesh_dir)
    segids = defaultdict(list)

    # pbar = tqmd()
    for entry in tqdm(os.scandir(mesh_dir)):
      # print(entry.name)
      pass
      # match = re.match(r'(\d+):(\d+).*.gz', entry.name)
      # if not match: 
      #   continue
      # fname, segid, lod = match.group(0), int(match.group(1)), int(match.group(2))
      # segids[segid].append(fname)

    
    
    # return segids, _info['mesh']
    return None, None


  # assert int(magnitude) == magnitude
  # start = 10 ** (magnitude - 1)
  # end = 10 ** magnitude

  # class FastMeshManifestTaskIterator(object):
  #   def __len__(self):
  #     return 10 ** magnitude
  #   def __iter__(self):
  #     for prefix in range(1, start):
  #       # print(layer_path)
  #       yield FastMeshManifestTask(layer_path=layer_path, prefix=str(prefix) + ':')

  #     # enumerate from e.g. 100 to 999
  #     for prefix in range(start, end):
  #       yield FastMeshManifestTask(layer_path=layer_path, prefix=prefix)

  # return FastMeshManifestTaskIterator()


class FastMeshManifestTask(object):
# def FastMeshManifestTask(RegisteredTask):
  # def __init__(self, layer_path, prefix, lod=0, mesh_dir=None):
    # super(FastMeshManifestTask, self).__init__(layer_path, prefix, lod, mesh_dir)
  #   super(FastMeshManifestTask, self).__init__(layer_path, prefix)
  #   self.layer_path = layer_path
  #   self.lod = lod
  #   self.prefix = prefix
  #   self.mesh_dir = mesh_dir
  def __init__(self, layer_path, segids, mesh_dir, lod=0):
    self.layer_path = layer_path
    self.segids = segids
    self.mesh_dir = mesh_dir
    self.lod = lod
  def execute(self):
    with Storage(self.layer_path) as storage:

      for segid, frags in tqdm(self.segids.items()):
        # file_path='{}/{}:{}'.format(self.mesh_dir, segid, self.lod)
        # logging.warning('fp: %s', file_path)
        storage.put_file(
            file_path='{}/{}:{}'.format(self.mesh_dir, segid, self.lod),
            content=json.dumps({"fragments": frags}),
            content_type='application/json',
        )
        a = 1

    # with Storage(self.layer_path) as storage:
    #   self._info = json.loads(storage.get_file('info').decode('utf8'))
    #   if self.mesh_dir is None and 'mesh' in self._info:
    #     self.mesh_dir = self._info['mesh']
    #   logging.warning('rank: %d, lp %s, pr %s', rank, self.layer_path, self.prefix)
      # self._generate_manifests(storage)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--labels', default=None,
                      help="path to precomputed labels")
  parser.add_argument('--verbose', default=False,
                      help="wether to use progressbar")

  args = parser.parse_args()
  if rank == 0:
    layer_path = 'file://'+ os.path.abspath(args.labels)
    print("Updating metadata...")
    segid_dict, mesh_dir = build_segid_map(layer_path)
    # for k, v in segid_dict.items():
    #   if len(v) > 3:
    #     print(k, len(v))
    # print(len(segid_dict.keys()))

    total_keys = list(segid_dict.keys())
    # tasks = tc.create_mesh_manifest_tasks(in_path, magnitude=3)
    # for t in tasks:
      # print(t)
    subset_keys = np.array_split(total_keys, size)
    segid_sub_dict = [{k:segid_dict[k] for k in k_set} for k_set in subset_keys ]
    # print([d.keys() for d in dicts])
    # sys.exit()

  else:
    layer_path = None
    segid_sub_dict = None
    mesh_dir = None
    # subset_keys = None
    # tasks = None
    # sub_ranges = None
  layer_path = comm.bcast(layer_path, 0)
  segid_sub_dict = comm.scatter(segid_sub_dict, 0)
  mesh_dir = comm.bcast(mesh_dir, 0)

  logging.warning('rank %d keys %d', rank, len(segid_sub_dict.keys()))
  fmmt = FastMeshManifestTask(layer_path, segid_sub_dict, mesh_dir)
  fmmt.execute()
  logging.warning('rank %d done', rank)
  comm.barrier()


  # tasks = comm.bcast(tasks, 0)
  # sub_ranges = comm.scatter(sub_ranges, 0)

  # print(len(sub_ranges))
  # mtq = mpiTaskQueue()
  # ltq = LocalTaskQueue(progress=True)
  # for i in sub_ranges:
  #   ltq.insert(tasks[i])
  # # ltq.run()
  # ltq._process(progress=True)
  # comm.barrier()
  # print('Done')












  # if rank == 0:
  #   in_path = 'file://'+args.labels
  #   print("Updating metadata...")
  #   mtq2 = mpiTaskQueue()
  #   tasks = tc.create_mesh_manifest_tasks(in_path, magnitude=3)
  #   # tasks = tc.create_mesh_manifest_tasks(in_path, magnitude=5)
  #   for t in tasks:
  #     mtq2.insert(t)
    
  #   L2 = len(mtq2._queue)
  #   print(len(mtq2._queue))
  #   all_range = np.arange(L2)
  #   sub_ranges = np.array_split(all_range, size)
  # else:
  #   sub_ranges = None
  #   mtq2 = None

  # sub_ranges = comm.bcast(sub_ranges, root=0)
  # mtq2 = comm.bcast(mtq2, root=0)
  # mtq2.run(sub_ranges[rank], args.verbose)
  # comm.barrier()


if __name__ == '__main__':
  main()
