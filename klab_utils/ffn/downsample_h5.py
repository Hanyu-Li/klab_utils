import h5py
from skimage import io
import argparse
import os
from os.path import exists, join, dirname, basename, abspath
import glob
import cv2
import re
import numpy as np
from tqdm import tqdm
from pprint import pprint
from klab_utils.aligntk import utils
from scipy import ndimage
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def mpi_downsample_h5(input, output, factor=(2,2,1), axes='zyx'):
  output_dir = dirname(abspath(output))
  os.makedirs(output_dir, exist_ok=True)

  if axes == 'zyx':
    factor = (factor[2], factor[1], factor[0])
    z_factor = factor[0]

  if mpi_rank == 0:
    with h5py.File(input, 'r') as f_in:
      in_shape = np.asarray(f_in['image'].shape)
      out_shape = in_shape // factor
      z_subsets = np.array_split(np.arange(out_shape[0]), mpi_size)
      print(in_shape, out_shape)
  else:
    z_subsets = None
    out_shape = None

  z_subsets = mpi_comm.scatter(z_subsets, 0)
  out_shape = mpi_comm.bcast(out_shape, 0)
  # print(mpi_rank)
  # print(z_subsets)

  with h5py.File(input, 'r', driver='mpio', comm=mpi_comm) as f_in:
    with h5py.File(output, 'w', driver='mpio', comm=mpi_comm) as f_out:
      img_ds = f_out.create_dataset('image', shape=out_shape, dtype=np.uint8)
      for z_out in tqdm(z_subsets, desc='z_slice'):
        # pass
        z_in = [z_out * z_factor + i for i in range(z_factor)]
        input_data = f_in['image'][z_in, :, :]
        zoom = 1.0 / np.asarray(factor)
        output_data = ndimage.zoom(input_data, zoom)
        print('>> %s %s %s' % (mpi_rank, input_data.shape, output_data.shape))
        img_ds[z_out,:,:] = output_data





def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--factor', default='2,2,1', type=str)
  args = parser.parse_args()
  factor = [int(i) for i in args.factor.split(',')]
  mpi_downsample_h5(args.input, args.output, factor)
if __name__ == '__main__':
  main()