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
        self.input_dir = input_dir
        self.output_dir = output_dir
        if self.output_dir == None:
            try:
                os.makedirs(os.path.join(self.input_dir, 'output'))
            except:
                pass
            self.output_dir = os.path.join(self.input_dir, 'output')
        self.flist = None
        
        self.MAX_ROW = 0
        self.MAX_COL = 0
        self.TILE_ROW = 0
        self.TILE_COL = 0
        self.TILE_MIN = 0
        self.TILE_MAX = 0
        self.DTYPE = 0
    def collect(self):
        collect_dir = os.path.join(self.output_dir, 'prealignment_stack')
        if not os.path.exists(collect_dir):
            os.makedirs(collect_dir)
        if self.stitch:
            flist = glob.glob(os.path.join(self.input_dir, 'S_*/img_t1_z1_c1'))
        else:
            flist = glob.glob(os.path.join(self.input_dir, 'S_*/*.tif'))
        for fname in tqdm(flist):
            #print(fname)
            im = cv2.imread(fname,flags=cv2.IMREAD_GRAYSCALE)
            #print(im.shape)
            RE = re.search('S_([0-9]*)', fname)
            ind = RE.group(1)
            #prefix_len = len(dataDir)

            #outf = outDir+fname[prefix_len:prefix_len+5]+'.tiff'
            outf = os.path.join(collect_dir, 'S_'+ind+'.tiff')
            #print(outf)
            tifffile.imsave(outf,im)

    def test_one_image(self):
        f_dummy = glob.glob(os.path.join(self.input_dir, 'S_*/Tile*.tif'))[0]
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
            for f in self.flist:
                tlist = glob.glob(os.path.join(f, 'Tile_*.tif'))
                if len(tlist) == 0:
                    continue

                for t in tlist:
                    res = re.search(r'Tile_r([0-9])-c([0-9])_S_([0-9]+)_*', t)     
                    #print(res.groups())
                    #print(res.group(1), res.group(2))      
                    r = res.group(1)
                    c = res.group(2)
                    z = int(res.group(3))

                    #command = t +'\t'+ c +'\t'+ r + '\t' + '0\t' + self.TILE_COL + '\t' + self.TILE_ROW + '\t' + '0\t'+ str(i)+'\n'
                    command = '{0} \t {1} \t {2} \t {3} \t {4} \t {5} \t {6} \t {7} \t {8} \n'.format(t,c,r,z,self.TILE_COL, self.TILE_ROW, self.TILE_MIN, self.TILE_MAX, self.DTYPE)
                    print(command)
                    output.write(command)

        
    def run(self):
        print("Input:", self.input_dir)
        print("Output:", self.output_dir)

    
        self.flist= glob.glob(os.path.join(self.input_dir,'S_*'))
        get_index = lambda f: int(f.split('/')[-1].split('_')[1])
        self.flist.sort(key=get_index)
        #pprint(self.flist)

        #self.collect()
        
        # Step 5: prepare TrackEM2 import txt file
        #self.prepare_align_txt(self.input_dir, self.input_dir)
        self.test_one_image()
        self.prepare_align_txt()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    rootDir = os.getcwd()
    #dataDir ='/home/vandana/Desktop/Section_Set_4/'
    emp = EM_preprocessor(args.input, args.output)
    emp.run()
    
    sys.exit()
    