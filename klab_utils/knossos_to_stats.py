#!/usr/bin/env python3
from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dxchange
#import snappy
import sys
import argparse
import os
from pprint import pprint
import re
import glob
import pandas as pd
import csv
from scipy.spatial import distance

#from PyQt5.QtWidgets import *
def extract_length(a):
    cumulative_len = 0
    e = a.getEdges()
    if not e:
        return cumulative_len
    for src_n in e.keys():
        s_coord = src_n.getCoordinate()
        for tgt_n in e[src_n]:
            t_coord = tgt_n.getCoordinate()
            cumulative_len = max(cumulative_len, distance.euclidean(s_coord,t_coord))
    return cumulative_len 
def sweep_sk(sk):
    result_dict = {}
    comments = [a.getComment() for a in sk.getAnnotations()]
    matches = [re.match(r"^(?P<category>[a-zA-Z]+)(?P<x>[0-9]+)($|[a-zA-Z_]+(?P<y>[0-9]+))", c).groupdict() for c in comments]
    #pprint(matches)
    for m in matches:
        cat = m['category']
        if cat not in result_dict.keys():
            result_dict[cat] = {'x':set(), 'y':set()}
        x = m['x']
        y = m['y']
        result_dict[cat]['x'].add(int(x))
        if y:
            result_dict[cat]['y'].add(int(y))
    for c in result_dict.keys():
        result_dict[c]['x'] = sorted(result_dict[c]['x'])
        result_dict[c]['y'] = sorted(result_dict[c]['y'])
    #pprint(result_dict)
    return result_dict

def get_stat(sk, result_dict):
    for cat in result_dict.keys():
        L_x = len(result_dict[cat]['x'])
        if not result_dict[cat]['y']:
            L_y = 1
        else:
            L_y = len(result_dict[cat]['y'])
        result_dict[cat]['data'] = np.zeros((L_x, L_y), dtype=np.float32)



    for a in sk.getAnnotations():
        euclidean_len = extract_length(a)
        c = a.getComment()
        if not c:
            continue
        m = re.match(r"^(?P<category>[a-zA-Z]+)(?P<x>[0-9]+)($|[a-zA-Z_]+(?P<y>[0-9]+))", c).groupdict()
        x = result_dict[m['category']]['x'].index(int(m['x']))
        if m['y']:
            y = result_dict[m['category']]['y'].index(int(m['y']))
        else:
            y = 0
        #print(m, euclidean_len, x, y)
        result_dict[m['category']]['data'][x,y] = euclidean_len
    return result_dict

def convert_to_csv(result_dict, f_out):
    
    for cat in result_dict.keys():
        x = result_dict[cat]['x']
        if result_dict[cat]['y']:
            y = result_dict[cat]['y']
        else:
            y = [0]
        ROW = pd.Index(x, name='Instance')
        COL = pd.Index(y, name='Z')
        df = pd.DataFrame(result_dict[cat]['data'], index=ROW, columns=COL)
        print(df)
        f_csv = os.path.join(f_out, cat+'.csv')
        print(f_csv)
        df.to_csv(f_csv)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', default='/mnt/md0/XRay/Knossos_measurements/WT_colon_380/knossos_cube/mag1/knossos.conf')
    parser.add_argument('--anno', default='/mnt/md0/XRay/Knossos_measurements/WT_colon_380/annotation/muscle_skeleton_annotation-20180113T1805.139.k.zip')
    args = parser.parse_args()
    print(args.cube, args.anno)
    root_dir = '.'
    f_knossos = args.cube
    f_overlay = args.anno
    f_out = os.path.join(root_dir, 'output')
    if not os.path.exists(f_out):
        os.makedirs(f_out)


    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(f_knossos)


    #raw = kd.from_raw_cubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])
    print(kd._experiment_name)
    print(kd.boundary)


    #overlay= kd.from_kzip_to_matrix(path=f_overlay,size=kd.boundary, offset=[0, 0, 0], mag=1, verbose=True, alt_exp_name_kzip_path_mode=False)
    sk = skeleton.Skeleton()
    sk.fromNml(f_overlay)
    result_dict = sweep_sk(sk)
    result_dict = get_stat(sk, result_dict)
    convert_to_csv(result_dict, f_out)
    sys.exit()

    #data_M = result_dict['M']['data']
    #data_SM = result_dict['SM']['data']
    #data_C = result_dict['C']['data']
    #data_CB = result_dict['CB']['data']


    
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(data_M)
    # plt.subplot(222)
    # plt.imshow(data_SM)
    # plt.subplot(223)
    # plt.plot(data_C)
    # plt.subplot(224)
    # plt.plot(data_CB)
    # plt.title('Cell Body Diameter')
    
    # data_M = data_M[:]
    # valid_M = data_M[data_M>0]

    # data_SM = data_SM[:]
    # valid_SM = data_SM[data_M>0]

    # data_C = data_C[:]
    # valid_C = data_C[data_C>0]

    # data_CB = data_CB[:]
    # valid_CB = data_CB[data_CB>0]
    # print(valid_CB.shape)
    # #l_M, hist_M = plt.hist(valid_M)

    # plt.figure()
    # plt.subplot(221)
    # plt.hist(valid_M, bins=100)
    # plt.subplot(222)
    # plt.hist(valid_SM, bins=100)
    # plt.subplot(223)
    # plt.hist(valid_, bins=100)
    # plt.subplot(224)
    # plt.hist(valid_M, bins=100)
    # plt.show()

    # sys.exit()
    # #sk2swc_and_center(sk, f_swc, f_center)

    # #print(overlay.shape)
    # #print(np.max(overlay[:]))
    # #overlay = np.rollaxis(overlay,2,0)
    # #overlay = np.rollaxis(overlay,2,1)
    # #dxchange.write_tiff(overlay.astype(np.uint32), 'test.tiff', overwrite=True)
if __name__ == "__main__":
    main()
