from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dxchange
#import snappy
import sys
import zipfile
import argparse
import os
import re
import glob
from PyQt5.QtWidgets import *

## Convert knossos traced dataset to: raw cube, annotation, and center pos
def sk2dense(sk, path):
    pass
def sk2swc(sk, path):
    with open(path, 'w') as trg_file:
        curr_id = 0
        for a in sk.getAnnotations():
            # after this, edges is a list of dictionaries containing the
            # edges: key: node, value: node obj
            #edges.append(a.getEdges())
            #rev_edges.append(a.getReverseEdges())
            #nodes = a.getNodes()
            #print(a)
            e = a.getEdges()
            #print(e.items())
            for src_n in e.keys():
                #print(src_n)
                src_id = src_n.getUniqueID()
                src_r = src_n.getDataElem('radius')
                if src_r > 1.5:
                    src_x = src_n.getCoordinate()[0]
                    src_y = src_n.getCoordinate()[1]
                    src_z = src_n.getCoordinate()[2]
                    n_str = '{0:d} {1:d} {2:f} {3:f} {4:f} {5:f} {6:d}'.format( \
                        src_id, curr_id, src_x, src_y, src_z, src_r, -1)
                    trg_file.write(n_str + '\n')
                if len(e[src_n]) > 0:
                    for trg_n in e[src_n]:
                        trg_x = trg_n.getCoordinate()[0]
                        trg_y = trg_n.getCoordinate()[1]
                        trg_z = trg_n.getCoordinate()[2]
                        trg_r = trg_n.getDataElem('radius')
                        trg_id = trg_n.getUniqueID()
                        n_str = '{0:d} {1:d} {2:f} {3:f} {4:f} {5:f} {6:d}'.format( \
                            trg_id, curr_id, trg_x, trg_y, trg_z, trg_r, src_id)
                        trg_file.write(n_str + '\n')
                curr_id += 1
def sk2swc_and_center(sk, swc_path, center_path):
    with open(swc_path, 'w') as trg_file, open(center_path, 'w') as aux_trg_file:
        curr_id = 0
        for a in sk.getAnnotations():
            # after this, edges is a list of dictionaries containing the
            # edges: key: node, value: node obj
            #edges.append(a.getEdges())
            #rev_edges.append(a.getReverseEdges())
            #nodes = a.getNodes()
            #print(a)
            e = a.getEdges()
            #print(e.items())
            for src_n in e.keys():
                #print(src_n)
                src_id = src_n.getUniqueID()
                src_r = src_n.getDataElem('radius')
                if src_r > 1.5:
                    src_x = src_n.getCoordinate()[0]
                    src_y = src_n.getCoordinate()[1]
                    src_z = src_n.getCoordinate()[2]
                    n_str = '{0:d} {1:d} {2:f} {3:f} {4:f} {5:f} {6:d}'.format( \
                        src_id, curr_id, src_x, src_y, src_z, src_r, -1)
                    trg_file.write(n_str + '\n')
                    aux_trg_file.write('{0:f} {1:f} {2:f}'.format(src_x, src_y, src_z)+ '\n')
                if len(e[src_n]) > 0:
                    for trg_n in e[src_n]:
                        trg_x = trg_n.getCoordinate()[0]
                        trg_y = trg_n.getCoordinate()[1]
                        trg_z = trg_n.getCoordinate()[2]
                        trg_r = trg_n.getDataElem('radius')
                        trg_id = trg_n.getUniqueID()
                        #n_str = '{0:d} 0 {1:f} {2:f} {3:f} {4:f} {' \
                        #        '5:d}'.format(
                        #    trg_id, trg_x, trg_y, trg_z, trg_r, src_id)
                        n_str = '{0:d} {1:d} {2:f} {3:f} {4:f} {5:f} {6:d}'.format( \
                            trg_id, curr_id, trg_x, trg_y, trg_z, trg_r, src_id)
                        trg_file.write(n_str + '\n')
                curr_id += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', default=None)
    parser.add_argument('--anno', default=None)
    args = parser.parse_args()
    f_knossos = args.cube
    f_overlay = args.anno
    f_out = './test_out.tif'
    matches = re.match('^.*/mag(?P<mag>[0-9]+)/.*$', args.cube).groupdict()
    mag = int(matches['mag'])

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(f_knossos)
    #raw = kd.from_raw_cubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])
    print(kd._experiment_name)
    print(kd.boundary)
    overlay= kd.from_kzip_to_matrix(path=f_overlay,size=kd.boundary/2, offset=[0, 0, 0], mag=mag, verbose=True, alt_exp_name_kzip_path_mode=True)
    #sk = skeleton.Skeleton()
    #sk.fromNml(f_overlay)
    #sk2swc_and_center(sk, f_swc, f_center)
    #sys.exit()
    print(overlay.shape)
    print(np.max(overlay[:]))
    overlay = np.rollaxis(overlay,2,0)
    overlay = np.rollaxis(overlay,2,1)
    #dxchange.write_tiff(overlay.astype(np.uint32), f_out, overwrite=True)
    dxchange.write_tiff(overlay.astype(np.uint8), f_out, overwrite=True)
if __name__ == '__main__':
    main()