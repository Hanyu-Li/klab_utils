"""Convert ffn inference result folder to a stack and precomputed volume."""
from ffn.inference import storage
import glob
import os
import numpy as np
import re
import argparse
from klab_utils.neuroglancer.vol_utils import write_vol
def get_zyx(fname):
  xyz = tuple([int(i) for i in re.search('seg-(\d+)_(\d+)_(\d+)\.npz', fname).groups()])
  zyx = (xyz[2], xyz[1], xyz[0])
  return zyx
def load_inference(input_dir):
  f = glob.glob(os.path.join(input_dir, '**/*.npz'), recursive=True)
  # f = [input_dir]
  # print('>>', f)
  #f = glob.glob(os.path.join(input_dir, '*.npz'))
  zyx = get_zyx(f[0])
  seg, _ = storage.load_segmentation(input_dir, zyx)
  return seg, np.array(zyx)

def convert_inference_to_stack(input_dir, output_dir, check_exist=True):
  seg, offset_zyx = load_inference(input_dir)
  print(seg.shape)
  seg = np.uint32(seg)
  size_zyx = seg.shape
  os.makedirs(output_dir, exist_ok=True)
  if check_exist:
    f = glob.glob(os.path.join(output_dir, '*.*'))
    if f:
      return offset_zyx, size_zyx
  write_vol(seg, output_dir, z_axis=0)
  return offset_zyx, size_zyx

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None)
  parser.add_argument('output', type=str, default=None)
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--mip', type=int, default=0)
  parser.add_argument('--mesh', action="store_true")

  args = parser.parse_args()
  stack_dir = os.path.join(args.output, 'stack')
  offset_zyx, size_zyx = convert_inference_to_stack(args.input, stack_dir)
  # z,y,x = zyx
  precomputed_dir = os.path.join(args.output, 'seg-%d_%d_%d_%d_%d_%d' % (
    offset_zyx[2], offset_zyx[1], offset_zyx[0],
    size_zyx[2], size_zyx[1], size_zyx[0]))



  # use external tools
  cmd_1 = 'klab_utils.neuroglancer.raw_to_precomputed_v2 \
    --label_dir %s\
    --output_dir %s\
    --resolution %s\
    --mip %d\
    --offset %d,%d,%d\
    --flip_xy' % (stack_dir, precomputed_dir, args.resolution, args.mip, x, y, z)
  
  cmd_2 = 'klab_utils.neuroglancer.mesh_generator_v2 \
      --labels %s \
      --verbose' % (os.path.join(precomputed_dir, 'segmentation'))
  os.system(cmd_1)
  if args.mesh:
    os.system(cmd_2)

if __name__ == '__main__':
  main()
