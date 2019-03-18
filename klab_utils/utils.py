import os
import argparse
import numpy as np
import cv2
import glob
from PIL import Image
from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def mpi_process(input_dir, output_dir, process_fn, params={}):
  '''MPI process

  Args:
    input_dir: input directory containing images
    output_dir: output directory
    process_fn: A function taking (img, **params) as input arguments
    params: additional parameters for process_fn
  '''
  if mpi_rank == 0:
    f_list = glob.glob(os.path.join(input_dir, '*.*'))
    f_list.sort()
    f_sublist = np.array_split(np.asarray(f_list), mpi_size)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
  else:
    f_sublist = None

  f_sublist = mpi_comm.scatter(f_sublist, 0)

  for f in tqdm(f_sublist):
    img = cv2.imread(f, 0)
    img_out = process_fn(img, **params)
    f_out = os.path.join(output_dir, os.path.basename(f))
    cv2.imwrite(f_out, img_out)