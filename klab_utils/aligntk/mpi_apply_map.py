import struct
from pprint import pprint
import glob
import os
import numpy as np
import argparse
import re
from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

  
def read_map(fn):
  """
  level = 1 << level downscale of grid
  x_min = post level (downscale) x_min position in image
  y_min ...
  width = post level (downscale)
  height = ...
  image = image that will be mapped to reference
  ref = image that used as ref
  elements = (x, y, c)
      x = x position in ref (post level)
      y = y position in ref (post level)
      c = confidence (0->1)
  elements is actually a 2d array with x changing fastest =
      [y0x0, y0x1, y0x2.... y(h-1)x(w-1)]
  """
  info = {}
  with open(fn, 'rb') as f:
    l = f.readline()
    #assert l.strip() == 'M1'
    l = f.readline()
    level = int(l.strip())
    l = f.readline()
    width, height = map(int, l.strip().split())
    l = f.readline()
    x_min, y_min = map(int, l.strip().split())
    l = f.readline()
    im_name, ref_name = l.strip().split()
    elements = []
    for _ in range(width * height):
      elements.append(struct.unpack('fff', f.read(12)))
    info['level'] = level
    info['width'] = width
    info['height'] = height
    info['x_min'] = x_min
    info['y_min'] = y_min
    info['elements'] = elements
    info['image'] = im_name
    info['ref'] = ref_name
  return info


def write_map(m, fn):
  assert 'level' in m
  assert 'elements' in m
  assert 'width' in m
  assert 'height' in m
  assert 'x_min' in m
  assert 'y_min' in m
  assert 'image' in m
  assert 'ref' in m
  assert len(m['elements']) == m['width'] * m['height']
  assert all([len(e) == 3 for e in m['elements']])
  with open(fn, 'wb') as f:
    # f.write('M1\n')
    f.write(bytearray("M1\n", 'ascii'))
    f.write(bytearray('%i\n' % m['level'], 'ascii'))
    f.write(bytearray('%i %i\n' % (m['width'], m['height']), 'ascii'))
    f.write(bytearray('%i %i\n' % (m['x_min'], m['y_min']), 'ascii'))
    f.write(bytearray('%s %s\n' % (m['image'], m['ref']), 'ascii'))
    for e in m['elements']:
      f.write(struct.pack('fff', *e))
def get_region(m):
  spacing = (1 << m['level']) * 1
  minX = 1000000000
  maxX = -1000000000
  minY = 1000000000
  maxY = -1000000000

  mh = m['height']
  mw = m['width']
  elem = m['elements']

  for y in range(0, mh-1):
    for x in range(0, mw-1):
      if elem[y * mw + x][2] == 0 or \
              elem[y * mw + x + 1][2] == 0.0 or \
              elem[(y+1) * mw + x][2] == 0.0 or \
              elem[(y+1) * mw + x + 1][2] == 0.0:
        continue
      for dy in range(0, 2):
        for dx in range(0, 2):
          rx = elem[(y + dy) * mw + x + dx][0] * spacing
          ry = elem[(y + dy) * mw + x + dx][1] * spacing
          if (rx < minX):
            minX = rx
          if (rx > maxX):
            maxX = rx
          if (ry < minY):
            minY = ry
          if (ry > maxY):
            maxY = ry
  return (minX, maxX, minY, maxY)

def get_global_region(amap_dir, sub_range):
  test_list = glob.glob(os.path.join(amap_dir, '*.map'))
  test_list.sort()
  ominX = 1000000000
  omaxX = -1000000000
  ominY = 1000000000
  omaxY = -1000000000
  for f_am in tqdm(test_list[sub_range[0]:sub_range[1]+1]):
    am = read_map(f_am)
    minX, maxX, minY, maxY = get_region(am)
    ominX = min(minX, ominX)
    omaxX = max(maxX, omaxX)
    ominY = min(minY, ominY)
    omaxY = max(maxY, omaxY)
  w = omaxX - ominX + 1
  h = omaxY - ominY + 1
  command = '%dx%d%+d%+d' % (w, h, ominX, ominY)
  return command

def get_subset_region(amap_subset):
  test_list = glob.glob(os.path.join(amap_dir, '*.map'))
  test_list.sort()
  ominX = 1000000000
  omaxX = -1000000000
  ominY = 1000000000
  omaxY = -1000000000
  for f_am in tqdm(test_list[690:700]):
    am = read_map(f_am)
    minX, maxX, minY, maxY = get_region(am)
    ominX = min(minX, ominX)
    omaxX = max(maxX, omaxX)
    ominY = min(minY, ominY)
    omaxY = max(maxY, omaxY)
  w = omaxX - ominX + 1
  h = omaxY - ominY + 1
  command = '%dx%d%d%d' % (w, h, ominX, ominY)
  return command

def _find_index(fname):
  res = re.search(r'([0-9]+)', fname)
  return int(res.group(1))

def write_tmp_images_lst(images_lst, groups, sub_range, outputs):
  with open(images_lst, 'r') as f:
    lines = f.readlines()

  # find sub_range
  lines_dict = {_find_index(l):l for l in lines}
  sub_range_lines = [lines_dict[i] for i in range(sub_range[0], sub_range[1]+1)]
  print(sub_range_lines)

  line_sets = np.array_split(np.asarray(sub_range_lines), groups)
  for i, ls in enumerate(line_sets):
    fname = os.path.join(outputs, 'images%d.lst' % i)
    with open(fname, 'w') as f1:
      f1.writelines(ls)
  
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str)
  parser.add_argument('--image_dir', type=str)
  parser.add_argument('--mask_dir', default=None, type=str)
  parser.add_argument('--image_lst', type=str)
  parser.add_argument('--sub_range', type=str, help='both inclusive')

  args = parser.parse_args()
  if mpi_rank == 0:
    tmp_dir = os.path.join(args.input, 'apply_map_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    sub_range = [int(i) for i in args.sub_range.split(',')]
    write_tmp_images_lst(args.image_lst, mpi_size, sub_range, tmp_dir)
    region = get_global_region(os.path.join(args.input, 'amaps'), sub_range)
    

    # test_list = glob.glob(os.path.join(amap_dir, '*.map'))
    # test_list.sort()
  else:
    tmp_dir = None
    region = None
  tmp_dir = mpi_comm.bcast(tmp_dir, 0)
  region = mpi_comm.bcast(region, 0)
  command = 'apply_map -image_list %s/images%d.lst -images %s -maps %s/amaps/ -output %s/aligned/ -masks %s -region %s -memory 500000' \
  % (tmp_dir, mpi_rank, args.image_dir, args.input, args.input, args.mask_dir, region )
  print(command)
  os.system(command)


  


