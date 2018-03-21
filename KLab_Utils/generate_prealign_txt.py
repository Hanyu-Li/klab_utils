from __future__ import print_function, division
import os
import glob
import tifffile
import numpy as np
from matplotlib.pyplot import *
import argparse
import shutil
import sys
import cv2
#import magic
import re
from PIL import Image
from ast import literal_eval as make_tuple
from tqdm import tqdm
from pprint import pprint
class EM_preprocessor(object):
    def __init__(self, input_dir, output_dir):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)

        if output_dir == None:
            output_dir = os.path.join(self.input_dir, 'output')
        try:
            os.makedirs(output_dir)
        except:
            pass
        self.output_dir = output_dir
        #self.flist = None
        self.flist = glob.glob(os.path.join(self.input_dir, '*.tif*'))
        #print(len(glob_list))
        #self.flist = glob_list
        #matches = re.match(r'^.*/.*(?P<index>[0-9]+).(tif*)$',self.input_dir)

        self.MAX_ROW = 0
        self.MAX_COL = 0
        self.TILE_ROW = 0
        self.TILE_COL = 0
        self.TILE_MIN = 0
        self.TILE_MAX = 0
        self.DTYPE = 0

    def test_one_image(self):
        #f_dummy = glob.glob(os.path.join(self.input_dir, 'S_*/Tile*.tif'))[0]
        print(len(self.flist))
        f_dummy = self.flist[0]
        dummy_data = cv2.imread(f_dummy,flags=cv2.IMREAD_GRAYSCALE)
        #dummy_data = tifffile.imread(f_dummy)
        #dummy_data = dummy_data[:,:,0]
        print(dummy_data.shape)
        self.TILE_ROW, self.TILE_COL = dummy_data.shape
        self.TILE_MIN, self.TILE_MAX = np.min(dummy_data[:]), np.max(dummy_data[:])
        print(self.TILE_ROW, self.TILE_COL, self.TILE_MIN, self.TILE_MAX, dummy_data.dtype)
        if dummy_data.dtype == np.uint8:
            print('8bit')
            self.DTYPE = 0
        elif dummy_data.dtype == np.uint16:
            print('16bit')
            self.DTYPE = 1



    def prepare_align_txt(self):
        #collect_dir = os.path.join(self.output_dir, 'prealignment_stack')
        f_align_txt = os.path.join(self.output_dir, 'align.txt')
        with open(f_align_txt,'w') as output:
            for i,f in enumerate(self.flist):
                command = '{}\t{}\t{}\t{}\n'.format(f, '0','0',str(i))
                print(command)
                output.write(command)
        
    def run(self):
        print("Input:", self.input_dir)
        print("Output:", self.output_dir)

    
        #self.flist= glob.glob(os.path.join(self.input_dir,'S_*'))
        #get_index = lambda f: int(f.split('/')[-1].split('_')[1])
        #self.flist.sort(key=get_index)
        #pprint(self.flist)

    
        # Step 5: prepare TrackEM2 import txt file
        #self.prepare_align_txt(self.input_dir, self.input_dir)
        self.test_one_image()
        self.prepare_align_txt()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='dir containing the stack, with *[id].tif')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    emp = EM_preprocessor(args.input, args.output)
    emp.run()
    
    sys.exit()
if __name__ == '__main__':
    main()
    
    