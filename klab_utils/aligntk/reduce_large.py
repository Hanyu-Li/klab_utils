import os
import argparse
import numpy as np
# import cv2
import glob
# from PIL import Image
import skimage
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from .utils import mpi_process
from .utils import reduce_size

def reduce_size(image, factor):
  img_out = cv2.resize(image, (0,0), fx=1.0/factor, fy=1.0/factor)
  img_out = skimage.tran
  X, Y = img_out.shape
  _X = X // 8 * 8
  _Y = Y // 8 * 8
  return img_out[:_X, :_Y]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--factor', default=2, type=int)
  parser.add_argument('--tile_mode', default=False, type=bool)
  args = parser.parse_args()

  mpi_process(args.input, args.output, reduce_size,  dict(factor=args.factor), tile_mode=args.tile_mode)



if __name__ == '__main__':
  main()
