import argparse
from .utils import mpi_process

def invert_background(image):
  image[image == 255] = 0
  return image
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  args = parser.parse_args()

  mpi_process(args.input, args.output, invert_background)

if __name__ == '__main__':
  main()
