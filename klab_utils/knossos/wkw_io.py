import wkcuber
import wkw
import zipfile
import wkw
from wkcuber.metadata import (
  refresh_metadata,
  detect_bbox,
  detect_layers,
  read_metadata_for_layer,
)
import subprocess
import argparse
import logging
import neuroglancer
from knossos_utils import skeleton
import zipfile
import glob
import os
from os.path import join, exists, basename
from klab_utils.neuroglancer import vol_utils
import untangle
import xmltodict
import multiprocessing
from pprint import pprint
import numpy as np
import xml.etree.ElementTree as ET
def read_downloaded_wkw(zip_path, datasource_path, axes='xyz'):
  """Return wkw segmentation in xyz shape"""
  
  target_dir = zip_path.rstrip('.zip')
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)
    
  if exists(join(target_dir, 'data.zip')):
    with zipfile.ZipFile(join(target_dir, 'data.zip'), 'r') as zip_ref:
      zip_ref.extractall(join(target_dir, 'data'))
      
  # assert nml and data exists
  nml_path = glob.glob(join(target_dir, '*.nml'))[0]
  seg_path = join(target_dir, 'data')
  assert exists(nml_path)
  assert exists(seg_path)
  mt = read_metadata_for_layer(datasource_path, 'color')
  bbox = mt[0]['boundingBox']
  off = bbox['topLeft']
  shape = (bbox['width'], bbox['height'], bbox['depth'])
  print(off)
  print(shape)
  scaling=(6,6,40)
  
  # read raw data 
  raw_ds = wkw.Dataset.open(join(datasource_path, 'color', '1'))
  raw = raw_ds.read(off, shape)
  
  ds = wkw.Dataset.open(join(seg_path, '1'))
  segmentation = ds.read(off, shape)
  print(np.mean(segmentation))
  if axes == 'xyz':
    raw = raw[0]
    seg =  segmentation[0]
  elif axes == 'zyx':
    raw = np.transpose(raw[0], [2,1,0])
    seg = np.transpose(segmentation[0], [2,1,0])
  return raw, seg, nml_path
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

def create_wkcube(fname, stack_path, wk_path, scale):
  cmd = r'python -m wkcuber --jobs %d --batch_size 32 \
    --layer_name color --max_mag 2 --scale %s \
    --name %s --no_compress --anisotropic %s %s' % (
      multiprocessing.cpu_count(),
      scale,
      fname, 
      stack_path, 
      wk_path)
  print(cmd)
  os.system(cmd)


def upload_segmentation(label_dir, output_dir, axes='zyx'):
  seg = vol_utils.get_stack_data(label_dir, axes=axes)
  upload_wkw(seg, output_dir, axes=axes)

def main():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--zip_file', type=str, default=None)
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None)
  parser.add_argument('--label_dir', default=None, type=str)
  parser.add_argument('--name', default=None, type=str)
  parser.add_argument('--scale', default='2,2,1', type=str)
  parser.add_argument('--resolution', default='6,6,40', type=str)

  # parser.add_argument('--datasource', default=None)
  args = parser.parse_args()

  if not args.name:
    name = 'test_cube'
  else:
    name = args.name

  create_wkcube(name, args.image_dir, args.output_dir, args.resolution)
  if args.label_dir:
    upload_segmentation(args.label_dir, args.output_dir, axes='zyx')
  