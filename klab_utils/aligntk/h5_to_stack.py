import h5py
import numpy as np
from skimage import io
import argparse
import os
import functools
from os.path import exists, join, dirname, basename, abspath
import time
from tqdm import tqdm
import multiprocessing
# from mpi4py import MPI
# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()
# mpi_size = mpi_comm.Get_size()

# def worker(h5_file, output_dir, z):
def worker(params):
  # print(h5_file, z)
  # print(params)
  with h5py.File(params['h5_file'], 'r') as f:

    slc = f['image'][params['z'], :]
    outname = os.path.join(params['output_dir'], 'S_%s.tif' % (str(params['z']).zfill(4)))
    # print(outname)
    # time.sleep(0.5)
    io.imsave(outname, slc)
    # print(slc.shape)
    return outname

def dummy(a):
  time.sleep(0.5)
  return a**2

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
#   parser.add_argument('--invert_first', action="store_true")
  args = parser.parse_args()
  n_cpu = multiprocessing.cpu_count()
  with h5py.File(args.input, 'r') as f:
    ds = f['image']
    z_indices = np.arange(ds.shape[0])
  pool = multiprocessing.Pool(n_cpu)
  input_params = [{'h5_file': args.input, 'output_dir': args.output, 'z': z} for z in z_indices]
  # results = pool.starmap(worker, tqdm(input_params))
  results = list(tqdm(pool.imap(worker, input_params), total=len(z_indices)))

  # results = list(tqdm(pool.imap(dummy, z_indices), total=len(z_indices)))

#   params = dict(invert_first=args.invert_first)

#   mpi_process(args.input, args.output, invert_background, params)
if __name__ == '__main__':
  main()