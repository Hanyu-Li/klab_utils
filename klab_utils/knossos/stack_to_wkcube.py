import cloudvolume
import numpy as np
from klab_utils.neuroglancer.vol_utils import get_cv_data, write_vol, clahe_vol
import os
import argparse
import logging

from collections import namedtuple
from wkcuber.cubing import cubing
from wkcuber.downsampling import downsample_mag, downsample, InterpolationModes
from wkcuber.mag import Mag
from wkcuber.metadata import write_webknossos_metadata
from wkcuber.utils import (
    open_wkw,
    WkwDatasetInfo,
)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source_path', default=None, type=str)
  parser.add_argument('target_path', default=None, type=str)
  parser.add_argument('--layer_name', default='color', type=str)
  parser.add_argument('--dtype', default='uint8', type=str)
  parser.add_argument('--scale', default=None, type=str)
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--distribution_strategy', default="multiprocessing", type=str)
  parser.add_argument('--jobs', default=1, type=int)
  parser.add_argument('--anisotropic', default=True, type=bool)
  #parser.add_argument('--anisotropic_target_mag', default='2-2-1', type=str)
  parser.add_argument('--mip', default=1, type=int)
  parser.add_argument('--buffer_cube_size', default=256, type=int)
  parser.add_argument('--interpolation_mode', default='default', type=str)
  parser.add_argument('--compress', default=False, type=bool)
  parser.add_argument('--name', default='test', type=str)
  parser.add_argument('--verbose', default=True, type=bool)
  #parser.add_argument('-v', dest='verbose', action='store_true')
  


  args = parser.parse_args()
	#args = ARGS(
	#	source_path=output_path,
	#	target_path=wk_path,
	#	layer_name='color',
	#	dtype='uint8',
	#	scale='6,6,40',
	#	batch_size=32,
	#	jobs=40,
	#	anisotropic_target_mag='2-2-1',
	#	mip=2,
	#	interpolation_mode='default',
	#	buffer_cube_size=256,
	#	compress=False,
	#	distribution_strategy='multiprocessing',
	#	name='hl007_subset_2'
	#)
	#print(args)
  if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
  os.makedirs(args.target_path, exist_ok=True)
  bbox = cubing(args.source_path, args.target_path, args.layer_name, args.dtype, args.batch_size, args)
  # anisotropic_target_mag = Mag(args.anisotropic_target_mag)
  # source_mag = Mag([1, 1, 1])
  #from_mag = Mag([1, 1, 1])
  # interpolation_mode = InterpolationModes.MEDIAN
  # buffer_cube_size = args.buffer_cube_size
  # for i in range(args.mip):
  #   target_mag = Mag([i * j for i, j in zip(anisotropic_target_mag.mag, source_mag.mag)])
  #   source_wkw_info = WkwDatasetInfo(args.target_path, args.layer_name, args.dtype, source_mag.to_layer_name())
  #   with open_wkw(source_wkw_info) as source:
  #       target_wkw_info = WkwDatasetInfo(
  #           args.target_path, args.layer_name, args.dtype, target_mag.to_layer_name()
  #       )
  #   buffer_cube_size //= (i+1) ** 2
  #   downsample(
  #       source_wkw_info,
  #       target_wkw_info,
  #       source_mag,
  #       target_mag,
  #       interpolation_mode,
  #       buffer_cube_size,
  #       args.compress,
  #       args
  #   )
  	
  scale = tuple(float(x) for x in args.scale.split(","))
  write_webknossos_metadata(
  				args.target_path,
  				args.name,
  				scale,
  				compute_max_id=False,
  				exact_bounding_box=bbox,
  		)


if __name__ == '__main__':
  main()
