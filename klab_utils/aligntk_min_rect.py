import os
import argparse
import numpy as np
import cv2
import imutils
import glob
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process


def get_min_rect(img, kernel, iterations):
  ret,thresh = cv2.threshold(img, 0.5, 255, 0)
  kernel = np.ones((kernel, kernel), np.uint8)
  thresh = cv2.dilate(thresh, kernel, iterations=iterations)
  _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  x, y, w, h = cv2.boundingRect(contours[0])
  return img[y:y+h, x:x+w]
# def get_thresh(img, kernel, iterations):
#   ret, thresh = cv2.threshold(img, 0.5, 255, 0)
#   kernel = np.ones((kernel, kernel), np.uint8)
#   thresh = cv2.dilate(thresh, kernel, iterations=iterations)
#   return thresh
# def get_min_rect(img, kernel, iterations):
#   thresh = get_thresh(img, kernel, iterations)

#   _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#   (x, y), (width, height), rect_angle = cv2.minAreaRect(contours[0])

#   angle = rect_angle
#   corrected_angle = 90 + rect_angle
# #   print(width, height, angle, corrected_angle)
#   if width < height:
#     angle = corrected_angle
#   else:
#     angle = angle
    
#   x, y, w, h = cv2.boundingRect(contours[0])
#   if angle < 1 and angle > -1 or corrected_angle < 1 and corrected_angle > -1:
#     return img[y:y+h, x:x+w]
#   else:
#     rot_img = imutils.rotate(img[y:y+h, x:x+w], angle)
#     rot_thresh = get_thresh(rot_img, kernel, iterations)
#     _, rot_contours, _ = cv2.findContours(rot_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     x, y, w, h = cv2.boundingRect(rot_contours[0])
#     return rot_img[y:y+h, x:x+w]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--kernel', default=3, type=int)
  parser.add_argument('--iterations', default=4, type=int)
  args = parser.parse_args()

  params = dict(kernel=args.kernel, iterations=args.iterations)
  mpi_process(args.input, args.output, get_min_rect, params)


if __name__ == '__main__':
  main()
