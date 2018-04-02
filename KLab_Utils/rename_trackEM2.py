import sys,os
import h5py
import dxchange
import argparse
import glob
import re

def rename_trackEM2(f_in_dir, f_out_dir=None):
    ''' rename trackEM2 output to match *[index].tif '''
    get_index = lambda f: int(re.match(r'^.*/.*_z(?P<index>[0-9]+).(.*)$', f).groupdict('index')['index'])
    print(f_in_dir)
    f_list = glob.glob(os.path.join(f_in_dir, '*.tif*'))
    for f in f_list:
        #print(f, get_index(f))
        new_name = os.path.join(f_in_dir, 'S_'+str(get_index(f)) +'.tif')
        print(new_name)
        os.rename(f,new_name)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='dir containing the stack, with *[id].0.tif')
    args = parser.parse_args()
    print(args.input)
    rename_trackEM2(args.input)

if __name__ == '__main__':
    main()
