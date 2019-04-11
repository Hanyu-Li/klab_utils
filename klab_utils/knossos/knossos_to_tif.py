from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# import dxchange
#import snappy
import sys
import zipfile
import argparse
import os
import re
import glob

# Convert knossos traced dataset to: raw cube, annotation, and center pos
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cube', default=None)
  parser.add_argument('--anno', default=None)
  parser.add_argument('--mag', type=int, default=1)
  parser.add_argument('--output', default=None)
  args = parser.parse_args()

  f_knossos = args.cube
  f_overlay = args.anno
  f_out = args.output
  # matches = re.match('^.*/mag(?P<mag>[0-9]+)/.*$', args.cube).groupdict()
  # mag = int(matches['mag'])
  mag = args.mag

  kd = knossosdataset.KnossosDataset()
  kd.initialize_from_knossos_path(f_knossos)
  #raw = kd.from_raw_cubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])
  print(kd._experiment_name)
  # sys.exit()

  print(kd.boundary)
  overlay = kd.from_kzip_to_matrix(path=f_overlay,
                                   #   size=kd.boundary//16,
                                   size=[1000, 1000, 1000],
                                   offset=[4300, 4300, 4300],
                                   mag=mag,
                                   verbose=True)
  #   alt_exp_name_kzip_path_mode=True)
  print(overlay.shape)
  print(np.max(overlay[:]))
  overlay = np.rollaxis(overlay, 2, 0)
  overlay = np.rollaxis(overlay, 2, 1)
  #dxchange.write_tiff(overlay.astype(np.uint32), f_out, overwrite=True)
  print(overlay.shape, overlay.dtype)
  print(np.mean(overlay[...]))

  # dxchange.write_tiff(overlay.astype(np.uint8), f_out, overwrite=True)
if __name__ == '__main__':
  main()
