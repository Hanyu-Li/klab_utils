from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dxchange
#import snappy
import sys
import zipfile
import argparse
import os
import glob

def offset_annotation(anno, dx, dy, dz):
  sk = skeleton.Skeleton()
  sk.fromNml(anno, use_file_scaling=True)
  f_out = os.path.join(os.path.dirname(anno), 'offset.k.zip')
  print(f_out)
  # print(sk.getAnnotations())
  # print(sk.annotation)
  for n in sk.getNodes():
    x, y, z = n.getCoordinate()
    #print('pre: {}, {}, {}'.format(x,y,z))
    n.setCoordinate((x+dx, y+dy, z+dz))
    #print('post: {}, {}, {}'.format(x+dx,y+dy,z+dz))

  sk.to_kzip(f_out, force_overwrite=True)

def main():
  pass
  parser = argparse.ArgumentParser()
  parser.add_argument('annotation', type=str)
  parser.add_argument('--dx', type=int, default=0)
  parser.add_argument('--dy', type=int, default=0)
  parser.add_argument('--dz', type=int, default=0)
  args = parser.parse_args()
  offset_annotation(args.annotation, args.dx, args.dy, args.dz)


if __name__ == '__main__':
  main()
