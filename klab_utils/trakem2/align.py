import os
import argparse
import numpy as np
import cv2
import re
import glob
from PIL import Image
from tqdm import tqdm
import logging
import pkg_resources



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--pairs', default=None, type=str)
  parser.add_argument('--min', default=1024, type=int)
  parser.add_argument('--max', default=2048, type=int)
  parser.add_argument('--begin', default=0, type=int)
  parser.add_argument('--fiji', default='fiji', type=str, help='specify ImageJ-linux64 executable path')
  args = parser.parse_args()

  os.makedirs(args.output, exist_ok=True)
  resource_path = 'ext/align.bsh'
  bsh_path = pkg_resources.resource_filename(__name__, resource_path)
  logging.warning('macro: %s', bsh_path)
  #split_aligntxt(args.input, args.output)

  #rank_input = os.path.join(args.output, 'align_%d.txt' % mpi_rank)

  if args.pairs:
    command = '%s --headless -Dinput=%s -Doutput=%s -Dpairs=%s -Dmin=%d -Dmax=%d -Dbegin=%d -- --no-splash %s' % (
        args.fiji, args.input, args.output, args.pairs, args.min, args.max, args.begin, bsh_path)
  else:
    command = '%s --headless -Dinput=%s -Doutput=%s -Dmin=%d -Dmax=%d -Dbegin=%d -- --no-splash %s' % (
        args.fiji, args.input, args.output, args.min, args.max, args.begin, bsh_path)
  print(command)
  os.system(command)

if __name__ == '__main__':
  main()
