import os
import argparse
import numpy as np
import cv2
import re
import glob
from PIL import Image
from tqdm import tqdm
from mpi4py import MPI
import logging
import pkg_resources
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def split_aligntxt(align_txt, output_dir):
  with open(align_txt, 'r') as f:
    lines = f.readlines()
  get_key = lambda x: int(re.search(r'S_(\d+)_', x).group(1))
  key_line_dict = {get_key(l):l for l in lines}
    
  key_set = np.asarray(list(key_line_dict.keys()))
  key_sublists = np.array_split(key_set, mpi_size)

  for i, keys in enumerate(key_sublists):
    with open(os.path.join(output_dir, 'align_%d.txt' % i), 'w') as f:
      for key in keys:
        f.writelines(key_line_dict[key])
def get_keys(align_txt):
  with open(align_txt, 'r') as f:
    lines = f.readlines()
  get_key = lambda x: int(re.search(r'S_(\d+).*', x).group(1))
  keys = np.asarray([get_key(l) for l in lines])
  keys.sort()
  key_sublists = np.array_split(keys, mpi_size)
  return key_sublists


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('output', default=None, type=str, help='A directory containing a project.xml file')
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--range', default=None, type=str)
  parser.add_argument('--heap_size', default=6, type=int)
  parser.add_argument('--fiji', default='fiji', type=str, help='specify ImageJ-linux64 executable path')
  args = parser.parse_args()

  # align stage


  # export stage
  if mpi_rank == 0:
    os.makedirs(args.output, exist_ok=True)
    resource_path = 'ext/export.bsh'
    bsh_path = pkg_resources.resource_filename(__name__, resource_path)
    logging.warning('macro: %s', bsh_path)
    # tmp_dir = os.path.join(args.output, 'tmp')
    # split_aligntxt(args.input, tmp_dir)
    if not args.range and args.input:
      key_sublist = get_keys(args.input)
    else:
      sub_range = [int(i) for i in args.range.split(',')]
      keys = np.asarray(range(sub_range[0], sub_range[1]))
      key_sublist = np.array_split(keys, mpi_size)

    if args.input:
      begin = get_keys(args.input)[0][0]
    else:
      begin = 0
  else:
    key_sublist = None
    bsh_path = None
    begin = None
  bsh_path = mpi_comm.bcast(bsh_path, 0)
  key_sublist = mpi_comm.scatter(key_sublist, 0)
  begin = mpi_comm.bcast(begin, 0)

  print(key_sublist)
  #rank_input = os.path.join(args.output, 'align_%d.txt' % mpi_rank)
  command = '%s -Xms%dg -Xmx%dg --headless -Dinput=%s -Doutput=%s -Drange=%s -Dbegin=%d -- --no-splash %s' % (
    args.fiji, args.heap_size, args.heap_size, args.input, args.output, '%d,%d' % (key_sublist[0], key_sublist[-1]), begin,
    bsh_path)
  print(command)
  os.system(command)

if __name__ == '__main__':
  main()
