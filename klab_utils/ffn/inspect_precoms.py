import neuroglancer
from ffn.inference import storage
from klab_utils.ffn.export_inference import load_inference, get_zyx
import numpy as np
import glob
import os
from os.path import join, dirname, basename
from ffn.utils import bounding_box
from pprint import pprint
import shutil
import re
import argparse
import neuroglancer
import numpy as np
from neuroglancer import url_state
import re

header = 'http://terminus.kasthurilab.com:9102/'
def show_link(root, prefix, local_mode):
  viewer = neuroglancer.Viewer()
  # out_root = '/home/hanyu/ng/HL007/precom_theta_full/region_3400_4360_0_2048_2048_512_v3/stage_1'
  stage_list = [i.strip(prefix) for i in glob.glob(os.path.join(root, 'precomputed*'))]
  seg_sources = [header+i for i in stage_list]
  # sk_source = header + '/HL007/precom_theta_full/region_3400_4360_0_2048_2048_512_v3/stage_1/skeletons_mip_0'
  # print(seg_sources)
  with viewer.txn() as s:
    # s.layers['image'] = neuroglancer.ImageLayer(
    #   source='precomputed://'+im_source
    # )
    for i, seg in enumerate(seg_sources[:5]):
  #     print(i, seg)
      s.layers['seg_%d' % i] = neuroglancer.SegmentationLayer(
        source='precomputed://%s' % seg
      )
  #   s.layers['sk'] = neuroglancer.SegmentationLayer(
  #       skeletons='precomputed://%s' % sk_source,
  #   )
    
  link = url_state.to_url(viewer.state)
  if local_mode:
    link = re.sub(r'https://neuroglancer-demo.appspot.com', r'http://localhost:8080', link)
  else:
    link = re.sub(r'https://', r'http://', link)
  pprint(link)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='a directory with precomputed-*, and config.pkl')
  parser.add_argument('--prefix', type=str, default='', 
    help='a directory with precomputed-*, and config.pkl')
  # parser.add_argument('--im_source', type=str, default='http://terminus.kasthurilab.com:9102/HL007/precomputed_1312/image', 
  #   help='a directory with precomputed-*, and config.pkl')
  # parser.add_argument('--header', type=str, default='http://terminus.kasthurilab.com:9102/', 
  #   help='a directory with precomputed-*, and config.pkl')
  parser.add_argument('--local', type=bool, default=False)
  args = parser.parse_args()
  show_link(args.input, args.prefix, args.local)

if __name__ == '__main__':
  main()

