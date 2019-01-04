#!/usr/bin/env python3
from __future__ import print_function, division
import os
import glob
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import sys
import cv2
#import magic
import re
from PIL import Image, ImageOps
from ast import literal_eval as make_tuple
from tqdm import tqdm
from pprint import pprint
class Fiji_Stitcher(object):
    def __init__(self, input_dir, output_dir, cmin, cmax, overlap):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cmin = cmin
        self.cmax = cmax
        self.overlap = overlap
        self.ijm_file = None
        self.flist = None
        self.cc_flist = None
        self.MIN_ROW = 0
        self.MIN_COL = 0
        self.MAX_ROW = 0
        self.MAX_COL = 0
        self.TILE_ROW = 0
        self.TILE_COL = 0

    def convert_to_8bit(self):
        #self.convert_ijm_file = os.path.join(self.output_dir, 'convert_test.ijm')
        #self.outdir = 
        ims = []
        cc_path = os.path.join(self.output_dir, 'cc')
        if not self.cc_flist:
            self.cc_flist = []
        if not os.path.exists(cc_path):
            os.makedirs(cc_path)

        for f_in in self.flist:
            fname = os.path.split(f_in)[-1]
            f_out = os.path.join(cc_path, fname)
            self.cc_flist.append(f_out)
            #print(out_f)
            im = cv2.imread(f_in,cv2.IMREAD_UNCHANGED)
            _im = np.clip(im,self.cmin, self.cmax)
            _im = np.uint8((_im-self.cmin) / (self.cmax-self.cmin) * 255)
            tifffile.imsave(f_out,_im)
        self.cc_flist = sorted(self.cc_flist)

    def prepare_macro(self):
        self.ijm_file = os.path.join(self.output_dir, 'test.ijm')
        commands = []
        with open(self.ijm_file,'w') as f:
            if self.cc_flist:
                FLIST = self.cc_flist
            else:
                FLIST = self.flist
            for fname in FLIST:
                matches = re.match(r"(?P<dirname>^.+)/(?P<exp_name>[^/]+)_y(?P<y>[0-9]+)_x(?P<x>[0-9]+).(?P<f_type>tif*)$", fname)
                print(matches.groupdict())
                row = int(matches['y'])
                col = int(matches['x'])
                dirname = matches['dirname']
                exp_name = matches['exp_name']
                f_type = matches['f_type']
                self.MIN_ROW = min(self.MIN_ROW, row)
                self.MIN_COL = min(self.MIN_COL, col)
                self.MAX_ROW = max(self.MAX_ROW, row)
                self.MAX_COL = max(self.MAX_COL, col)
            print(self.MAX_ROW, self.MAX_COL)
            #self.MAX_ROW = max(row)
            #self.MAX_COL = max(col)
            #longer_col = self.MAX_ROW < self.MAX_COL
            #min_ind = np.argmin(col if longer_col else row)
            #max_ind = np.argmax(col if longer_col else row)
            #continue
            #tilename = tiles[0][0:6]+'{y}'+tiles[0][7:9]+'{x}'+tiles[0][10:]
            tilename = exp_name+'_y{y}_x{x}'+'.'+f_type
            print(tilename)
            #command = 'run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x='+str(self.MAX_COL)+' grid_size_y='+str(self.MAX_ROW)+ \
            #' tile_overlap=8 first_file_index_y='+str(self.MIN_COL)+' first_file_index_x='+str(self.MIN_ROW)+' directory='+dirname+ \
            #' file_names='+tilename+' output_textfile_name=TileConfiguration.txt fusion_method=[Average] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory='+self.output_dir+'");\n'
            # more pythonic

            command = 'run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x={0:d} grid_size_y={1:d}  \
            tile_overlap={2:d} first_file_index_y={3:d} first_file_index_x={4:d} directory={5:s} \
            file_names={6:s} output_textfile_name=TileConfiguration.txt fusion_method=[{7:s}] regression_threshold={8:.2f} max/avg_displacement_threshold={9:.2f} \
            absolute_displacement_threshold={10:.2f} compute_overlap {11:s} computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory={12:s}");\n'.format(\
            self.MAX_COL,\
            self.MAX_ROW,\
            self.overlap,\
            self.MIN_COL,\
            self.MIN_ROW,\
            dirname,\
            tilename,\
            'Linear Blending',\
            0.30,\
            2.50,\
            3.50,\
            'subpixel_accuracy',\
            self.output_dir)

            #make sure you have the correct tile_overlap value and correct x and y units for the grid size
            print(command)
            #print(command)
            f.write(command)
            #min_fname = os.path.join(dirname, tiles[min_ind])
            #max_fname = os.path.join(dirname, tiles[max_ind])
            #print(min_fname,max_fname)

                

                    
                    
                #except:
                    #continue
        pass
    def run_fiji(self):
        print('fiji --headless -macro '+self.ijm_file)
        os.system('fiji --headless -macro '+self.ijm_file)
        pass
                    



             

                
                
                
    def unify_size(self, x_cut=0, y_cut=0):
        get_index = lambda f: int(f.split('/')[-1].split('_')[1].split('.')[0])
        out_flist = glob.glob(os.path.join(self.output_dir, 'S_*.tif'))
        out_flist.sort(key=get_index)

        cut_out_dir = os.path.join(self.output_dir, 'cut')
        #print(out_flist)
        
        first_flag = True
        for o in out_flist[0:3]:
            old_data = cv2.imread(o,flags=cv2.IMREAD_GRAYSCALE)
            print(old_data.shape)
            if first_flag:
                new_x = (x_cut, old_data.shape[0]-x_cut)
                new_y = (y_cut, old_data.shape[0]-y_cut)
                print(new_x,new_y)
                first_flag = False
            cv2.imwrite(os.path.join(self.output_dir,output_fname), new_data)
            break

    def run(self):
        print("Input:", self.input_dir)
        print("Output:", self.output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.flist= glob.glob(os.path.join(self.input_dir,'*.tif*'))
        #get_index = lambda f: int(f.split('/')[-1].split('_')[1])
        #self.flist.sort(key=get_index)
        self.flist.sort()
        #pprint(self.flist)
        
        #matches = [re.match(r"^(?P<category>[a-zA-Z]+)(?P<x>[0-9]+)($|[a-zA-Z_]+(?P<y>[0-9]+))", c).groupdict() for c in comments]
        # Step 1: Prepare stitch macro, fix large image
        self.convert_to_8bit()
        self.prepare_macro()
        
        # Step 2: Run stitch
        self.run_fiji()
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--cmin', type=float)
    parser.add_argument('--cmax', type=float)
    parser.add_argument('--overlap', type=int)
    #parser.add_argument('--stitch', default=False, type=bool)
    args = parser.parse_args()
    #rootDir = os.getcwd()
    fs = Fiji_Stitcher(args.input, args.output, args.cmin, args.cmax, args.overlap)
    fs.run()
if __name__ == '__main__':
    main()
    