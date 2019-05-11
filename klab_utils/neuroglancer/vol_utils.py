import cloudvolume
import h5py
import numpy as np
import cv2
import h5py
import glob
import re
import os
from skimage import io
from tqdm import tqdm

def clahe_vol(vol, clipLimit=2.0, tileGridSize=(8,8)):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  new_vol = [clahe.apply(vol[:,:,i]) for i in tqdm(range(vol.shape[2]))]
  new_vol = np.stack(new_vol, -1)
  return new_vol


def get_h5_data(h5_paths, flip=True):
  ref_data = []
  for h5_path in h5_paths:
    f_path, dataset_name = h5_path.split(':')
    with h5py.File(f_path, 'r') as f:
      ref_data.append(f[dataset_name][...])
      
  ref_data =  np.concatenate(ref_data, 0)
  if flip:
    ref_data = np.transpose(ref_data, [2,1,0])
  return ref_data

def get_stack_data(image_dir):
  f_list = glob.glob(os.path.join(image_dir, '*.*'))
#   print(f_list)
  get_ind = lambda f: int(re.search(r'(\d+)\..*', f).group(1))
  f_list.sort(key=get_ind)
  vol = np.stack([io.imread(f) for f in f_list], -1)
  return vol

def get_cv_data(cv_path, offset_xyz, size_xyz):
  full_cv = cloudvolume.CloudVolume('file://%s' % cv_path, mip=0, parallel=True)
                    
  bbox = [cloudvolume.Bbox(i, i + j) for i,j in zip(offset_xyz, size_xyz)]
  return np.concatenate([full_cv[b] for b in bbox], 2)[...,0]


def write_vol(vol, output_dir, z_axis=2, start_ind=0):
  os.makedirs(output_dir, exist_ok=True)
  if z_axis != 2:
    vol = np.moveaxis(vol, z_axis, 2)
  for i in range(vol.shape[2]):
    fname = os.path.join(output_dir, 'S_%s.tif' % str(i+start_ind).zfill(4))
    io.imsave(fname, vol[:,:,i])
