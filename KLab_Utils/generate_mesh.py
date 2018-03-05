import json
import os.path
import shutil
import argparse

import numpy as np
from cloudvolume import Storage, CloudVolume, EmptyVolumeException
import cloudvolume.lib as lib
from taskqueue import MockTaskQueue

from igneous import (
    DownsampleTask, MeshTask, MeshManifestTask, 
    QuantizeAffinitiesTask, HyperSquareConsensusTask,
    DeleteTask
)
from igneous import downsample
from igneous.task_creation import create_downsample_scales, create_downsampling_tasks, create_quantized_affinity_info

#def delete_layer(path=layer_path):
#    if os.path.exists(path):
#        shutil.rmtree(path)  
def test_mesh_manifests(directory, mesh_dir):
    layer_path = 'file://' + directory
    to_path = lambda filename: os.path.join(directory, mesh_dir, filename)


    n_segids = 600
    n_lods = 2
    n_fragids = 5

    #with Storage(layer_path) as stor:
    #    stor.put_file('info', '{"mesh":"mesh_mip_3_error_40"}'.encode('utf8'))

    for segid in range(n_segids):
        for lod in range(n_lods):
            for fragid in range(n_fragids):
                filename = '{}:{}:{}'.format(segid, lod, fragid)
                lib.touch(to_path(filename))

    for i in range(10):
        MeshManifestTask(layer_path=layer_path, prefix=i, lod=0).execute()

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
            assert os.path.exists(to_path(filename))
            filename = '{}:1'.format(segid)
            #assert not os.path.exists(to_path(filename))

    for i in range(10):
        MeshManifestTask(layer_path=layer_path, prefix=i, lod=1).execute()

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
        assert os.path.exists(to_path(filename))
        filename = '{}:1'.format(segid)
        assert os.path.exists(to_path(filename))

    with open(to_path('50:0'), 'r') as f:
        content = json.loads(f.read())
        assert content == {"fragments": [ "50:0:0","50:0:1","50:0:2","50:0:3","50:0:4" ]}

    #if os.path.exists(directory):
    #    shutil.rmtree(directory)
        
        
def test_mesh(layer_path):
    #delete_layer()
    #storage, _ = create_layer(size=(64,64,64,1), offset=(0,0,0), layer_type="segmentation")
    storage = Storage(layer_path)
    cv = CloudVolume(storage.layer_path)
    print(cv.volume_size)
    # create a box of ones surrounded by zeroes
    #data = np.zeros(shape=(64,64,64,1), dtype=np.uint32)
    #data[1:-1,1:-1,1:-1,:] = 1
    #cv[0:64,0:64,0:64] = data

    t = MeshTask(
        shape=cv.volume_size,
        offset=(0,0,0),
        layer_path=storage.layer_path,
        mip=0,
    )
    t.execute()
    assert storage.get_file('mesh/1:0:0-64_0-64_0-64') is not None 
    print( list(storage.list_files('mesh/')) )
    #assert list(storage.list_files('mesh/')) == ['mesh/1:0:0-64_0-64_0-64']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'layer', default=None)
    args = parser.parse_args()

    #directory = '/mnt/md0/KLab_Util_test/mesh_test/new_labels/'
    layer_path = 'file://'+args.layer
    test_mesh_manifests(args.layer, 'mesh')
    test_mesh(layer_path)
    #delete_layer(layer_path)

if __name__ == '__main__':
    main()