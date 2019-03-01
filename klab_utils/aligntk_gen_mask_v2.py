import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def gen_mask(img, l_thresh, h_thresh, kernel_size):
  kernel = np.ones((kernel_size,kernel_size),np.uint8)
  mask = 255*(np.logical_or(img < l_thresh,img > h_thresh)).astype(np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.dilate(mask,kernel,iterations = 3)
  mask = 255 - mask
  return mask


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--lower_threshold', default=5, type=int)
  parser.add_argument('--higher_threshold', default=250, type=int)
  parser.add_argument('--kernel', default=10, type=int)
  args = parser.parse_args()

  f = args.input
  f_mask = args.output
  print(f, f_mask)

  bname = os.path.basename(f).split('.')[0]
  img = cv2.imread(f, 0)
  mask = gen_mask(img, args.lower_threshold, args.higher_threshold, args.kernel)
  with open(f_mask, 'wb') as fd:
    row, col = mask.shape
    fd.write(bytearray("P4\n%i %i\n" % (col, row), 'ascii'))
    fd.write(np.packbits(mask[:row,:col], axis=-1).tobytes())

if __name__ == '__main__':
  main()
