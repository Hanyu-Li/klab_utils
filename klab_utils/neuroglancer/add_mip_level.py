import cloudvolume
import numpy as np
import argparse

def add_mip(vol_path, factor):
  c_path = 'file://%s' % vol_path
  vol = cloudvolume.CloudVolume(c_path, parallel=True)
  max_mip = len(vol.info['scales'])
  #new_vol = cloudvolume.CloudVolume('file://%s' % vol_path, mip=max_mip, parallel=True)
  new_info = vol.add_scale([i**max_mip for i in factor])
  print(vol.info)
  vol.commit_info()
  
  from_vol = cloudvolume.CloudVolume(c_path, mip=max_mip-1, parallel=True, progress=True)
  np_vol = np.asarray(from_vol[::factor[0], ::factor[1], ::factor[2]])
  to_vol = cloudvolume.CloudVolume(c_path, mip=max_mip, parallel=True, progress=True, info=vol.info)
  to_vol[:,:,:] = np_vol
  pass

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, help='path to cloud volume')
  parser.add_argument('--factor', type=str, default='2,2,1', help='downsample factor')
  args = parser.parse_args()

  factor = [int(i) for i in args.factor.split(',')]
  add_mip(args.input, factor)

if __name__ == '__main__':
  main()
