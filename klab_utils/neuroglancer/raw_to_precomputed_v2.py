'''Converts a image stack to cloudvolume '''
from __future__ import print_function
#from memory_profiler import profile
import logging
import numpy as np
import sys
import glob
# from cv2 import imread
from skimage import io
import sys
import argparse
import os
import re
from cloudvolume import CloudVolume
from pprint import pprint

from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

logger = logging.getLogger(__name__)

def _find_index(fname):
  res = re.search(r'([0-9]+)\.(\.*)', fname)
  return int(res.group(1))


def mpi_cloud_write(f_sublist, c_path, start_z, mip, factor, chunk_size, flip_xy, cv_args):
  ''' f_sublist is a List[List[str]], inner list size == z_batch'''
  for fb in tqdm(f_sublist):
    if flip_xy:
      loaded_vol = np.stack([np.transpose(io.imread(f)) for f in tqdm(fb, desc='loading')], axis=2)
    else:
      loaded_vol = np.stack([io.imread(f) for f in tqdm(fb, 'loading')], axis=2)
    diff = chunk_size[2] - loaded_vol.shape[2]
    if diff > 0:
      loaded_vol = np.pad(loaded_vol, ((0,0), (0,0), (0, diff)), 'constant', constant_values=0)

    curr_z = _find_index(fb[0])
    actual_z = curr_z - start_z
    for m in range(mip+1):
      cv = CloudVolume(c_path, mip=m, **cv_args)
      offset = cv.mip_voxel_offset(m)
      step = np.array(factor)**m
      cv_z_start = actual_z // step[2] + offset[2]

      # diff = chunk_size[2] - loaded_vol.shape[2]
      # if diff > 0:
      #   loaded_vol = np.pad(loaded_vol, ((0,0), (0,0), (0, diff)), 'constant', constant_values=0)

      # cv_z_size = loaded_vol.shape[2] // step[2]
      cv_z_size = loaded_vol.shape[2]
      logging.warn('mip %d, writing %s %s', m, cv_z_start, cv_z_size)
      # cv_z_size = loaded_vol.shape[2] 

      # if cv_z_size < chunk_size[2]:
      #   cv[:, :, cv_z_start:cv_z_start + chunk_size[2]] = 0
      cv[:, :, cv_z_start:cv_z_start + cv_z_size] = loaded_vol
      loaded_vol = loaded_vol[::factor[0], ::factor[1], ::factor[2]]
    del loaded_vol
  return


def divide_work(f_list, num_proc, z_batch, image_size, memory_limit):
  '''Decide the best way to distribute workload and z_batch size.
      min granularity is z_batch
  '''
  batched_f_list = np.split(np.asarray(
      f_list), np.arange(z_batch, len(f_list), z_batch))
  if num_proc > len(batched_f_list):
    print('>>', num_proc, len(batched_f_list))
    filled_num_proc = len(batched_f_list)
  else:
    filled_num_proc = num_proc

  f_sublists = np.array_split(batched_f_list, filled_num_proc)
  for _ in range(num_proc - filled_num_proc):
    f_sublists.append([])
  print('>>>', len(f_sublists))

  return f_sublists


def stack_to_cloudvolume(input_dir, output_dir, layer_type, mip,
                         chunk_size, offset, resolution, factor, flip_xy, memory_limit=10000):
  '''Converts a stack of images to cloudvolume.'''
  if mpi_rank == 0:
    f_list = glob.glob(os.path.join(input_dir, '*.*'))
    f_list.sort(key=_find_index)
    os.makedirs(output_dir, exist_ok=True)
    im0 = io.imread(f_list[0])
    if flip_xy:
      im0 = np.transpose(im0)
    print(im0.shape, im0.dtype)
    data_type = im0.dtype
    Z = len(f_list)
    X, Y = im0.shape
    start_z = _find_index(f_list[0])
    print('start_z:', start_z)
    im_size = np.float32(sys.getsizeof(im0)) / 1073741824
    print('single_image_size: \t %.4f GB' % im_size)
    print('total_size: \t\t %.4f GB' % (im_size * Z))
    print('mpi_proc: \t\t %d' % mpi_size)
    #print('total_limit: \t\t %.4f GB' % memory_limit)
    print('peak memory: \t\t %.4f GB' %
          (im_size * chunk_size[2] * mpi_size))

    f_sublist = divide_work(
        f_list, mpi_size, chunk_size[2], im_size, memory_limit)
    c_path = 'file://' + os.path.join(output_dir, layer_type)
    # factor = [2, 2, 1]

    if layer_type == 'image':
      encoding = 'raw'
      compress = False
    elif layer_type == 'segmentation':
      encoding = 'compressed_segmentation'
      compress = True

    pad = lambda x,i: ((x-1) // i + 1) * i
    pad_X, pad_Y, pad_Z = pad(X, chunk_size[0]), pad(Y, chunk_size[1]), pad(Z, chunk_size[2])
    print(X, Y, Z)
    print(pad_X, pad_Y, pad_Z)
 

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=str(data_type),
        encoding=encoding,
        resolution=list(resolution),
        voxel_offset=np.array(offset),
        volume_size=[X, Y, pad_Z],
        # volume_size=[X, Y, Z],
        chunk_size=chunk_size,
        max_mip=mip,
        factor=factor,
        # compressed_segmentation_block_size=compressed_segmentation_block_size,
    )
    cv_args = dict(
        bounded=True, fill_missing=True, autocrop=False,
        cache=False, compress_cache=None, cdn_cache=False,
        progress=False, info=info, provenance=None, compress=compress, 
        non_aligned_writes=True, parallel=1)
    cv = CloudVolume(c_path, mip=0, **cv_args)

    # correct a bug in cloudvolume
    if encoding == 'compressed_segmentation':
      for i in range(1, mip+1):
        info['scales'][i]['compressed_segmentation_block_size'] = info['scales'][0]['compressed_segmentation_block_size']

    pprint(cv.info)
    cv.commit_info()
    # pprint(cv.info)

  else:
    f_sublist = None
    c_path = None
    start_z = None
    cv_args = None
    factor = None
  f_sublist = mpi_comm.scatter(f_sublist, root=0)
  c_path = mpi_comm.bcast(c_path, root=0)
  start_z = mpi_comm.bcast(start_z, root=0)
  cv_args = mpi_comm.bcast(cv_args, root=0)
  factor = mpi_comm.bcast(factor, root=0)

  if not len(f_sublist):
    return
  print('rank: %d, range: %s <-> %s' % (
      mpi_rank, f_sublist[0][0], f_sublist[-1][-1]))
  mpi_cloud_write(f_sublist, c_path, start_z, mip,
                  factor, chunk_size, flip_xy, cv_args)
  return

# @profile


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', default=None)
  parser.add_argument('--label_dir', default=None)
  parser.add_argument('--output_dir', default='./precomputed')
  parser.add_argument('--resolution', type=str, default='10,10,10')
  parser.add_argument('--mip', type=int, default=3)
  parser.add_argument('--chunk_size', type=str, default='64,64,64')
  parser.add_argument('--z_step', type=int, default=None)
  parser.add_argument('--factor', type=str, default='2,2,1')
  parser.add_argument('--flip_xy', action="store_true")
  parser.add_argument('--memory_limit', type=float, default=10000)
  parser.add_argument('--offset', type=str, default='0,0,0')


  args = parser.parse_args()
  resolution = tuple(int(d) for d in args.resolution.split(','))
  chunk_size = tuple(int(d) for d in args.chunk_size.split(','))
  offset = tuple(int(d) for d in args.offset.split(','))
  factor = tuple(int(d) for d in args.factor.split(','))

  if args.image_dir:
    stack_to_cloudvolume(
      args.image_dir, 
      args.output_dir,
      layer_type='image', #data_type='uint8',
      mip=args.mip,
      chunk_size=chunk_size,
      offset=offset,
      resolution=resolution,
      factor=factor,
      flip_xy=args.flip_xy,
      memory_limit=args.memory_limit)
  if args.label_dir:
    stack_to_cloudvolume(
      args.label_dir, 
      args.output_dir,
      layer_type='segmentation', #data_type='uint32',
      mip=args.mip,
      chunk_size=chunk_size,
      offset=offset,
      resolution=resolution,
      factor=factor,
      flip_xy=args.flip_xy,
      memory_limit=args.memory_limit)


if __name__ == '__main__':
  main()
