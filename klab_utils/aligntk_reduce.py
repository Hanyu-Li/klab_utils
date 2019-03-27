import os
import argparse
import numpy as np
import cv2
import glob
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process
# from mpi4py import MPI
# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()
# mpi_size = mpi_comm.Get_size()
def reduce_size(image, factor):
  img_out = cv2.resize(image, (0,0), fx=1.0/factor, fy=1.0/factor)
  X, Y = img_out.shape
  _X = X // 8 * 8
  _Y = Y // 8 * 8
  return img_out[:_X, :_Y]
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--factor', default=2, type=int)
  args = parser.parse_args()

  mpi_process(args.input, args.output, reduce_size,  dict(factor=args.factor))



  # if mpi_rank == 0:
  #   f_list = glob.glob(os.path.join(args.input, '*.*'))
  #   f_list.sort()
  #   f_sublist = np.array_split(np.asarray(f_list), mpi_size)
  #   #img = cv2.imread(args.input, 0)
  #   #X, Y = img.shape
  # else:
  #   f_sublist = None
  #   #X, Y = None, None
  # f_sublist = mpi_comm.scatter(f_sublist, 0)
  # # X = mpi_comm.bcast(X, 0)
  # # Y = mpi_comm.bcast(Y, 0)

  # for f in tqdm(f_sublist):
  #   img = cv2.imread(f, 0)
  #   img_out = reduce(img, args.factor)
  #   f_out = os.path.join(args.output, os.path.basename(f))
  #   cv2.imwrite(f_out, img_out)


if __name__ == '__main__':
  main()
