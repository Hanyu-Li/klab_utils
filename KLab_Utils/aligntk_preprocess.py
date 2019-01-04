import os
import argparse
import glob 


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', default=None, type=str)
  parser.add_argument('--output_dir', default=None, type=str)
  args = parser.parse_args()
  f_images = os.path.join(args.output_dir, 'images.lst')
  f_pairs = os.path.join(args.output_dir, 'pairs.lst')
  image_list = [s.split('.')[0] for s in glob.glob1(args.image_dir, '*.tif*')]
  image_list.sort()
  with open(f_images, 'w') as f1:
    [f1.write(im+'\n') for im in image_list]

  with open(f_pairs, 'w') as f2:
    for i in range(len(image_list)-1):
      f2.write(image_list[i] + ' ' + image_list[i+1] + ' ' 
        + image_list[i] + '_' + image_list[i+1] + '\n')


if __name__ == '__main__':
  main()