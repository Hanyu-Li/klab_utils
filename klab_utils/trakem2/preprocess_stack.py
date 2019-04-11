"""Preprocess a directory with S_0000.tif, S_0001.tif, ... S_*.tif into trakem2 import text file"""
from __future__ import print_function, division
import os
import glob
import tifffile
import numpy as np
from matplotlib.pyplot import *
import argparse
import shutil
import sys
import cv2
#import magic
import re
from PIL import Image
from ast import literal_eval as make_tuple
from tqdm import tqdm
from pprint import pprint
class EM_preprocessor(object):
  def __init__(self, input_dir, output):
    self.input_dir = input_dir
    self.output = output
    output_dir = os.path.dirname(self.output)
    os.makedirs(output_dir, exist_ok=True)

    self.flist = None
    self.MAX_ROW = 0
    self.MAX_COL = 0

    self.TILE_ROW = 0
    self.TILE_COL = 0
    self.TILE_MIN = 0
    self.TILE_MAX = 0
    self.DTYPE = 0

  def test_one_image(self):
    f_dummy = glob.glob(os.path.join(self.input_dir, '*.tif'))[0]
    dummy_data = cv2.imread(f_dummy, flags=cv2.IMREAD_GRAYSCALE)
    print(dummy_data.shape)
    self.TILE_ROW, self.TILE_COL = dummy_data.shape
    self.TILE_MIN, self.TILE_MAX = np.min(dummy_data[:]), np.max(dummy_data[:])
    print(self.TILE_ROW, self.TILE_COL, self.TILE_MIN,
          self.TILE_MAX, dummy_data.dtype)
    if dummy_data.dtype == np.uint8:
      print('8bit')
      self.DTYPE = 0
    elif dummy_data.dtype == np.uint16:
      print('16bit')
      self.DTYPE = 1

  def prepare_align_txt(self):
    #f_align_txt = os.path.join(self.output_dir, 'align.txt')

    flist = np.asarray(glob.glob(os.path.abspath(
        os.path.join(self.input_dir, '*.tif*'))))
    inds = [int(re.search('.*_([0-9]*)', f.split('/')[-1]).group(1))
            for f in flist]
    flist = flist[np.argsort(inds)]
    with open(self.output, 'w') as output:
      for i, f in enumerate(flist):
        command = '{0} \t {1} \t {2} \t {3} \t {4} \t {5} \t {6} \t {7} \t {8} \n'.format(
            f, 0, 0, i, self.TILE_COL, self.TILE_ROW, self.TILE_MIN, self.TILE_MAX, self.DTYPE)
        print(command)
        output.write(command)
      output.close()

  def run(self):
    print("Input:", self.input_dir)
    print("Output:", self.output)

    self.flist = glob.glob(os.path.join(self.input_dir, 'S_*'))
    pprint(self.flist)
    self.test_one_image()
    self.prepare_align_txt()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  args = parser.parse_args()
  emp = EM_preprocessor(args.input, args.output)
  emp.run()


if __name__ == '__main__':
  main()
