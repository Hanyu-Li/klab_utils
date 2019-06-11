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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--zip_file', type=str, default=None)
  parser.add_argument('--datasource', default=None)
  parser.add_argument('--output_dir', default='./precomputed')