import os
import argparse
import glob 
import numpy as np
import cv2
from PIL import Image
import netpbmfile

def gen_mask(img, threshold):
  mask = (img < threshold).astype(np.uint8)
  return mask


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', default=None, type=str)
  parser.add_argument('--mask_dir', default=None, type=str)
  parser.add_argument('--threshold', default=1, type=float)
  args = parser.parse_args()
  for f in glob.glob(os.path.join(args.image_dir, '*.tif*')):
    bname = os.path.basename(f).split('.')[0]
    f_mask = os.path.join(args.mask_dir, bname+'.pbm')
    img = cv2.imread(f, 0)
    print(f_mask)
    mask = gen_mask(img, args.threshold)
    print(mask.dtype)

    with open(f_mask, 'wb') as fd:
      row, col = mask.shape
      # row, col = 19720, 24976
      print(row, col)
      fd.write(bytearray("P4\n%i %i\n" % (col, row), 'ascii'))
      fd.write(np.packbits(mask[:row,:col], axis=-1).tobytes())


if __name__ == '__main__':
  main()