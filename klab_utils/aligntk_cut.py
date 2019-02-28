import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--start_xy', default='0,0', type=str)
  parser.add_argument('--size_xy', default='1024,1024', type=str)
  parser.add_argument('--angle', default=0.0, type=float, 
    help='counter clockwise degrees')
  args = parser.parse_args()

  sx, sy  = [int(i) for i in args.start_xy.split(',')]
  dx, dy =  [int(i) for i in args.size_xy.split(',')]
  print(sx, sy, dx, dy)
  dx = dx // 8 * 8
  dy = dy // 8 * 8

  img = cv2.imread(args.input, 0)
  X, Y = img.shape
  rot_mat = cv2.getRotationMatrix2D((Y/2, X/2), args.angle, 1)
  img = cv2.warpAffine(img, rot_mat, (Y, X))
  cv2.imwrite(args.output, img[sx:sx+dx, sy:sy+dy])


if __name__ == '__main__':
  main()
