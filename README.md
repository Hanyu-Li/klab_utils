# klab_utils

A toolbox of scripts facilitating connectomics data processing. Interfaces with a few popular frameworks like Knossos, TrackEM2, neuroglancer and Cloud-Volume

# Installing

1. Knossos: 
``` bash
pip install git+https://github.com/knossos-project/knossos_utils.git
pip install git+https://github.com/knossos-project/knossos_cuber.git
```
2. Neuroglancer, follow building guide: https://github.com/google/neuroglancer
and install the python wheel
``` bash
cd {neuroglancer_dir}/python
python setup.py bundle_client
pip install -e .
```

3. Cloud Volume(for converting raw data into neuroglancer compatible forms)
``` bash
pip install git+https://github.com/seung-lab/cloud-volume.git
```

4. Igneous(for 3D mesh generation)
``` bash
pip install git+https://github.com/seung-lab/igneous.git
```

5. Dxchange(for tiff stack io):
``` bash
conda install dxchange
```

6. Install
``` bash

pip install -e .
```