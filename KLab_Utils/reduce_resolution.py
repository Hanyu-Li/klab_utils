'''Reduce Resolution of EM images.'''
import os
import argparse
import cv2
import glob 
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default=None, type=str)
  parser.add_argument('--output_dir', default=None, type=str)
  parser.add_argument('--factor', default=2, type=int)
  args = parser.parse_args()

  downsample_ratio = 1.0 / args.factor
  tiles = glob.glob(os.path.join(args.input_dir, 'S_*/*.tif*'))
  tiles.sort()
  print('Number of Tiles: ', len(tiles))
  print('Downsample Ratio: ', downsample_ratio)

  for t in tqdm(tiles):
    dirname = os.path.basename(os.path.dirname(t))
    filename = os.path.basename(t)
    os.makedirs(os.path.join(args.output_dir, dirname), exist_ok=True)
    f_out = os.path.join(args.output_dir, dirname, filename)
    im = cv2.imread(t, 0)
    im_down = cv2.resize(im, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
    cv2.imwrite(f_out, im_down)

if __name__ == '__main__':
  main()