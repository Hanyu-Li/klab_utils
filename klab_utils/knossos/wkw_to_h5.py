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
import multiprocessing
import numpy as np
import h5py

def read_downloaded_wkw(
  zip_path, 
  datasource_path, 
  axes='xyz',
  resolution=(6,6,40)):
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
  # scaling=(6,6,40)
  
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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('zip_path', default=None, type=str)
  parser.add_argument('datasource_path', default=None)
  parser.add_argument('--output_path', default=None)
  parser.add_argument('--resolution', default='6,6,40', type=str)

  args = parser.parse_args()
  resolution=[int(i) for i in args.resolution.split(',')]

  raw, seg, nml_path = read_downloaded_wkw(
    args.zip_path, 
    args.datasource_path, 
    # args.output_dir, 
    resolution=resolution)
  
  with h5py.File(args.output_path, 'w') as f:
    f['image'] = raw
    f['label'] = seg
  

