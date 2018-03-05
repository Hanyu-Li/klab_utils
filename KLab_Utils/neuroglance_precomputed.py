#!/usr/bin/env python3

from __future__ import print_function

import webbrowser
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import neuroglancer
import dxchange
import argparse
import os, signal
import subprocess
import re
from .reader import omni_read

def glance_precomputed(viewer, image=None,host='localhost', labels=None, port=41000):
    '''supply cloud paths'''
    if image: 
        image_path = 'precomputed://http://localhost:{}/'.format(port)+os.path.relpath(image)
        #image_path = 'precomputed://'+image
        #image_path = 'precomputed://ftp://localhost/cloud_test'
        #image_path = 'precomputed://http://localhost:41000'
        print('Image Source:', image_path)
    else:
        return None
    if labels: 
        labels_path = 'precomputed://http://localhost:{}/'.format(port)+os.path.relpath(labels)
        print('Labels Source:',labels_path)
    with viewer.txn() as s:
        if image is not None:
            s.layers['image'] = neuroglancer.ImageLayer(
                source=image_path
            )
        if labels is not None:
            s.layers['ground_truth'] = neuroglancer.SegmentationLayer(
                source=labels_path
            )
    return viewer.get_viewer_url()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'precomputed', default=None, help='relative path in the file server')
    parser.add_argument( '--server_port', type=int, default=41000 )
    parser.add_argument( '--client_port', type=int, default=42000 )
    args = parser.parse_args()

    neuroglancer.set_server_bind_address(bind_address='127.0.0.1', bind_port=args.client_port)
    viewer = neuroglancer.Viewer()
    def dummy():
        print('hello callback')
    #viewer.defer_callback(dummy)
    image_path = os.path.join(args.precomputed , 'image')
    labels_path = os.path.join(args.precomputed , 'labels')

    #subprocess.call('http-server --cors -p '+str(args.server_port), shell=False)
    #os.system('http-server --headless -macro '+self.ijm_file)
    url = glance_precomputed(viewer=viewer, image=image_path, labels=labels_path, port=args.server_port)
    print(url)
    webbrowser.open_new_tab(url)


    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()

if __name__ == '__main__':
    main()