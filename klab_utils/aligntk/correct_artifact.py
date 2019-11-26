import numpy as np
import glob
import os
from pprint import pprint
import shutil
import re
import numpy as np
import cv2
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm, tqdm_notebook
import skimage
from scipy import ndimage
import scipy
import argparse
import shutil
from .contrast_adjust import ecdf, hist_norm
from .utils import mpi_process
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None, type=str)
  parser.add_argument('--downsample', default=1, type=int)
  parser.add_argument('--sigma', default=20, type=int)
  parser.add_argument('--var_thresh', default=0.4, type=float)
  parser.add_argument('--pearson_thresh', default=0.5, type=float)
  parser.add_argument('--min_patch_size', default=900, type=float)
  parser.add_argument('--large_patch_thresh', default=50000, type=float)
  args = parser.parse_args()
  return args

def get_fg_mask(img, threshold=10):
  mask = np.greater_equal(img, threshold)
  labels, _ = ndimage.label(mask)
  ids, counts = np.unique(labels, return_counts=True)
  max_id = ids[1:][np.argmax(counts[1:])]
  fg_mask = labels == max_id
  return fg_mask
      

def get_outliers(
  raw_image, 
  ref_image, 
  downsample=10, 
  sigma=50,
  var_thresh=0.5,
  pearson_thresh=0.2,
  min_patch_size=100,
  large_patch_thresh=50000,
):
  """Detect outlier type and treat accordingly.
  For each outlier patch, hist norm to global,
  if beyond salvation, turn to black 
  
  Args:
    raw_image: input image to process
    ref_image: previous image in the aligned stack, to pinpoint anomalies
    downsample: downample ratio
    sigma: blur kernel sigma
    var_thresh: if mean value of region deviate more than var_thresh * std_val
      considered candidate outlier
    pearson_thresh: if an outlier patch has higher pearson coef than threshold
      it will not be considered an outlier, but simply a regional shift of 
      brightness(e.g. soma)
    min_patch_size: above which to be considered as a valid patch, ignore tiny
      dusts
    large_patch_thresh: above which is considered a 'large' patch whose patch stats
      subject to a larger region of disruption, we'll attempt hist_norm large patches
      nontheless
    
  Returns:
    output image
  """
  im = raw_image[::downsample, ::downsample]
  ref_im = ref_image[::downsample, ::downsample]
  
  # detect high deviation from global mean std
  mask = get_fg_mask(im, 10)
  ref_mask = get_fg_mask(ref_im, 10)
  
  mean_val = np.mean(im[mask].ravel())
  std_val = np.std(im[mask].ravel())
  
  # compute image wise skew
  global_skew = scipy.stats.skew(im[mask].ravel())
  
  # filter image for less noisy pearson corr calculation
  kernel = np.ones((sigma, sigma),np.float32)/ sigma**2
  im_filtered = cv2.filter2D(im.astype(np.float32),-1,kernel)
  im_filtered[np.logical_not(mask)] = 0
  ref_im_filtered = cv2.filter2D(ref_im.astype(np.float32),-1,kernel)
  ref_im_filtered[np.logical_not(ref_mask)] = 0
  
  outliers = np.logical_and(
    np.logical_or(im_filtered > (mean_val + var_thresh * std_val),
                  im_filtered < (mean_val - var_thresh * std_val)),
    np.logical_and(mask, ref_mask)).astype(np.uint8)
  
  # deal with each patch separately in outliers
  patches, ids = ndimage.label(outliers)
  
  patch_type_dict = {} # type 1: hist_norm fix, type 2: clear as background
  for i in range(1,ids+1):
    patch_mask = patches == i
    if np.sum(patch_mask) < min_patch_size:
      outliers[patch_mask] = 0
      continue
    
#     patch_inds = np.flatnonzero(patches.ravel() == i)
    patch_inds = np.flatnonzero(patch_mask.ravel())
    curr_raw_patch = im.ravel()[patch_inds]
    ref_raw_patch = ref_im.ravel()[patch_inds]
    curr_fil_patch = im_filtered.ravel()[patch_inds]
    ref_fil_patch = ref_im_filtered.ravel()[patch_inds]
    
    curr_std = np.std(curr_fil_patch)
    
    score = pearsonr(curr_fil_patch, ref_fil_patch)[0]
    mean_diff = np.abs(np.mean(curr_fil_patch) - np.mean(ref_fil_patch))
          
    if not score or score > pearson_thresh or mean_diff < var_thresh*std_val * 0.5:
      outliers[patch_mask] = 0
    else:
      # skew metric check
      patch_skew = scipy.stats.skew(curr_raw_patch, bias=False)
      ref_skew = scipy.stats.skew(ref_raw_patch, bias=False)
      global_skew_diff = np.abs(global_skew - patch_skew)
      ref_skew_diff = np.abs(ref_skew - patch_skew)
      # print('skews: ', i, global_skew_diff, ref_skew_diff)
      if ref_skew_diff < 0.4:
        outliers[patch_mask] = 0
      elif global_skew_diff > 0.6 and np.sum(patch_mask[:]) < large_patch_thresh:
        outliers[patch_mask] = 2
        patch_type_dict[i] = 2
      else:
        patch_type_dict[i] = 1
#       print('id: %d, skew: %.2f, global_skew: %.2f, ref_skew: %.2f, mean: %.2f, std: %.2f, size %d' % (
#         i, patch_skew, global_skew, ref_skew, np.mean(im[patch_mask]), np.std(im[patch_mask]), np.sum(patch_mask[:])))
        
#   output_im = output_im_flat.reshape(output_im.shape)
#   output_im[outliers==2] = 0
  patches[outliers==0] = 0
  
  outliers = skimage.transform.rescale(outliers, downsample, 
    anti_aliasing=True, preserve_range=True, multichannel=False, order=0)
  patches = skimage.transform.rescale(patches, downsample, 
    anti_aliasing=True, preserve_range=True, multichannel=False, order=0)

  return outliers, patches, patch_type_dict

def fix_outliers(im, patches, patch_type_dict):
  """Fix a image with outliers and patches."""
#   print(patches.shape)
  mask = get_fg_mask(im, 10)
  
  output_im = im.copy()
  output_im[np.logical_not(mask)] = 0
  output_im_flat = output_im.ravel()
  
  xg, yg = ecdf(im[mask].ravel())
  bin_edges = np.arange(0, 255, 5)
  quantiles = []
  for b in bin_edges:
    q = np.where(xg>b)[0]
    if len(q) > 1:
      quantiles.append(yg[q[0]])
    else:
      quantiles.append(1.0)
      
#   patch_ids = np.unique()
  for pid, ptype in patch_type_dict.items():
    patch_mask = patches==pid
    patch_inds = np.flatnonzero(patch_mask.ravel())
    if ptype == 1:
      patch_im = im.ravel()[patch_inds]
      remap_patch_im = hist_norm(patch_im, bin_edges, quantiles) 
      output_im_flat[patch_inds] = remap_patch_im
    elif ptype == 2:
      output_im_flat[patch_inds] = 0
      
  output_im = output_im_flat.reshape(im.shape)
#   output_im[outliers==2] = 0
  return output_im
      
  
  
def mpi_run(input_dir, output_dir, **params):
  if mpi_rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    flist = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    # print(flist)
    flist_pairs = list(zip(flist[1:], flist[:-1]))
    flist_pairs_subset = np.array_split(flist_pairs, mpi_size)
    # print(flist_pairs_subset[1])
  # if mpi_rank == 0:
  else:
    flist_pairs_subset = None

  flist_pairs_subset = mpi_comm.scatter(flist_pairs_subset, 0)

  # print(flist_pairs_subset)
  for fp in tqdm(flist_pairs_subset):
    # print(fp[0], fp[1])
    raw_image = skimage.io.imread(fp[0])
    ref_image = skimage.io.imread(fp[1])
    fout = os.path.join(output_dir, os.path.basename(fp[0]))
    _, patches, patch_type_dict = get_outliers(
      raw_image,
      ref_image,
      **params
    )
    out_image = fix_outliers(raw_image, patches, patch_type_dict)
    # print('pre post:', np.mean(raw_image), np.mean(out_image))
    # print(fout)
    skimage.io.imsave(fout, out_image)
  
  if mpi_rank == 0:
    # copy over the first im
    # fin = flist[0]
    fout = os.path.join(output_dir, os.path.basename(flist[0]))
    shutil.copy(flist[0], fout)
    # print('rank 0', fout)
  mpi_comm.barrier()

  
def main():
  args = parse_args()
  # params = vars(args)
  # params = dict(
  #   downsample=args.downsample,
  #   sigma=20,
  #   var_thresh=0.4,
  #   pearson_thresh=0.3,
  #   min_patch_size=900,
  # )
  # dummy_run(**vars(args))
  mpi_run(**vars(args))
  # mpi_run(args.input_dir, args.output_dir, params)
  # mpi_run(args.input, args.output, **vars(args))


if __name__ == '__main__':
  main()