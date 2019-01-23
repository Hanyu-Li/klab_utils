'''Reduce Resolution of EM images.'''
import os
import argparse
import cv2
import glob 
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default=None, type=str)
  parser.add_argument('--output_dir', default='ds', type=str)
  parser.add_argument('--size', default=1024, type=int)
  parser.add_argument('--offset', default=None, type=str)
  args = parser.parse_args()



  tiles = glob.glob(os.path.join(args.input_dir, 'S_*/*.tif*'))
  tiles.sort()

  
  if args.offset == None:
    im0 = cv2.imread(tiles[0], 0)
    offset = im0.shape()   / 2

  print('Number of Tiles: ', len(tiles))
  print('Preview Size: ', args.size)
  print('Preview offset', offset)

  for t in tqdm(tiles):
    dirname = os.path.basename(os.path.dirname(t))
    filename = os.path.basename(t)
    os.makedirs(os.path.join(args.output_dir, dirname), exist_ok=True)
    f_out = os.path.join(args.output_dir, dirname, filename)
    im = cv2.imread(t, 0)
    im_down = im[offset[0]:offset[0]+args.size,offset[1]:offset[1]+args.size]
    cv2.imwrite(f_out, im_down)

if __name__ == '__main__':
  main()