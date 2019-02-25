import os
import argparse
import glob 
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def gen_mask(img, l_thresh, h_thresh, kernel_size):
  kernel = np.ones((kernel_size,kernel_size),np.uint8)
  mask = 255*(np.logical_or(img < l_thresh,img > h_thresh)).astype(np.uint8)
  mask = 255 - mask
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.dilate(mask,kernel,iterations = 3)
  return mask


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', default=None, type=str)
  parser.add_argument('--mask_dir', default='masks', type=str)
  parser.add_argument('--lower_threshold', default=5, type=float)
  parser.add_argument('--higher_threshold', default=250, type=float)
  parser.add_argument('--kernel', default=10, type=float)
  args = parser.parse_args()
  os.makedirs(args.mask_dir, exist_ok=True)

  for f in tqdm(glob.glob(os.path.join(args.image_dir, '*.tif*'))):
    bname = os.path.basename(f).split('.')[0]
    f_mask = os.path.join(args.mask_dir, bname+'.pbm')
    img = cv2.imread(f, 0)
    mask = gen_mask(img, args.lower_threshold, args.higher_threshold, args.kernel)
    with open(f_mask, 'wb') as fd:
      row, col = mask.shape
      fd.write(bytearray("P4\n%i %i\n" % (col, row), 'ascii'))
      fd.write(np.packbits(mask[:row,:col], axis=-1).tobytes())


if __name__ == '__main__':
  main()
