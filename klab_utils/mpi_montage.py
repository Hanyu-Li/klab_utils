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
    text = f.readlines()
  get_key = lambda x: int(re.search(r'S_(\d+)_', x).group(1))
  # keys = [get_key(line) for line in text]
  key_line_dict = {}
  for line in text:
    key = get_key(line)
    val = key_line_dict.get(key, [])
    val.append(line)
    key_line_dict[key] = val
    
  key_set = np.asarray(list(key_line_dict.keys()))
  key_sublists = np.array_split(key_set, mpi_size)
  # print(key_sublists)

  for i, keys in enumerate(key_sublists):
    with open(os.path.join(output_dir, 'align_%d.txt' % i), 'w') as f:
      for key in keys:
        f.writelines(key_line_dict[key])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  args = parser.parse_args()

  if mpi_rank == 0:
    os.makedirs(args.output, exist_ok=True)
    resource_path = 'ext/montage_macro.bsh'
    bsh_path = pkg_resources.resource_filename(__name__, resource_path)
    logging.warning('macro: %s', bsh_path)
    split_aligntxt(args.input, args.output)
    # print(bsh_path)
  else:
    pass
    bsh_path = None
  bsh_path = mpi_comm.bcast(bsh_path, 0)

  rank_input = os.path.join(args.output, 'align_%d.txt' % mpi_rank)
  command = 'fiji --headless -Dinput=%s -Doutput=%s -- --no-splash %s' % (
    rank_input, args.output, bsh_path)
  print(command)
  os.system(command)

if __name__ == '__main__':
  main()