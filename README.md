# klab_utils

A toolbox of scripts facilitating connectomics data processing. Interfaces with a few popular frameworks like Knossos, TrackEM2, neuroglancer and Cloud-Volume

# Installing

``` bash
sudo apt-get install build-essential libboost-dev
pip install -r requirements.txt
pip install -e .
```

(optional) Neuroglancer, follow building guide: https://github.com/google/neuroglancer
and install the python wheel
``` bash
cd {neuroglancer_dir}/python
python setup.py bundle_client
pip install -e .
```