
'''Build wkw dataset with cubing and segmentation layer ready for incremental tracing.'''
import wkw
import zipfile
from wkcuber.metadata import (
  refresh_metadata,
  detect_bbox,
  detect_layers,
  read_metadata_for_layer,
)

import argparse
import logging
import glob
import os
from os.path import join, exists
import re
import multiprocessing
import numpy as np
from skimage.io import imread

def read_vol(image_dir, axes='zyx', n_image=None):
  f_list = glob.glob(os.path.join(image_dir, '*.*'))
  get_ind = lambda f: int(re.search(r'(\d+)\..*', f).group(1))
  f_list.sort(key=get_ind)
  f_list = np.array(f_list)
  if n_image is None:
    ranges = np.arange(len(f_list))
  else:
    ranges = np.arange(n_image)

  if axes == 'zyx':
    vol = np.stack([np.squeeze(imread(f)) for f in f_list[ranges]], 0)
  elif axes == 'yxz':
    vol = np.stack([np.squeeze(imread(f)) for f in f_list[ranges]], -1)
  elif axes == 'xyz':
    vol = np.stack([np.squeeze(imread(f)) for f in f_list[ranges]], 0)
    vol = np.transpose(vol, [2, 1, 0])

  return vol

def upload_wkw(seg, datasource_path, axes='zyx'):
  """Upload segmentation to datasource_path, and refresh datasource-properties.json."""
  # write 
  ds_path = join(datasource_path, 'segmentation', '1')
  os.makedirs(ds_path, exist_ok=True)
  if axes == 'zyx':
    seg = np.transpose(seg, [2,1,0])
  elif axes == 'xyz':
    pass
  else:
    raise ValueError('axes has to be xyz or zyx')
    
#   try:
  with wkw.Dataset.create(ds_path, wkw.Header(np.uint32)) as ds:
    ds.write([0,0,0], np.uint32(seg))
#   except:
#     ValueError('Already Written')
  
  mt = read_metadata_for_layer(datasource_path, 'color')
  bbox = mt[0]['boundingBox']
  
  refresh_metadata(datasource_path, compute_max_id=True, exact_bounding_box=bbox)

def create_wkcube(fname, stack_path, wk_path, max_mag, scale, wkcuber=None):
  cmd = ''
  if wkcuber:
    cmd = cmd + r'cd %s ; ' % wkcuber 
  cmd = cmd + r'python -m wkcuber --jobs %d --batch_size 1 --layer_name color --max_mag %d --scale %s --name %s --no_compress %s %s' % (
      multiprocessing.cpu_count(),
      max_mag,
      scale,
      fname, 
      stack_path, 
      wk_path)
  print(cmd)
  os.system(cmd)


def upload_segmentation(label_dir, output_dir, axes='zyx'):
  # seg = get_stack_data(label_dir, axes=axes)
  seg = read_vol(label_dir, axes=axes)
  upload_wkw(seg, output_dir, axes=axes)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None)
  parser.add_argument('--label_dir', default=None, type=str)
  parser.add_argument('--name', default=None, type=str)
  # parser.add_argument('--scale', default='2,2,1', type=str)
  parser.add_argument('--max_mag', default=2, type=int)
  parser.add_argument('--resolution', default='6,6,40', type=str)

  args = parser.parse_args()

  if not args.name:
    name = 'test_cube'
  else:
    name = args.name

  #create_wkcube(name, args.image_dir, args.output_dir, args.max_mag, args.resolution)
  if args.label_dir:
    upload_segmentation(args.label_dir, args.output_dir, axes='zyx')
  
