import sys
from setuptools import setup

entry_points = {
    'console_scripts': [
        'knossos_to_stats = KLab_Utils.knossos_to_stats:main',
        'knossos_to_tif = KLab_Utils.knossos_to_tif:main',
        'fiji_stitch = KLab_Utils.fiji_stitch:main',
        'neuroglance_raw = KLab_Utils.neuroglance_raw:main',
        'neuroglance_precomputed = KLab_Utils.neuroglance_precomputed:main',
        'neuroglance_public = KLab_Utils.neuroglance_public:main',
        'raw_to_precomputed = KLab_Utils.raw_to_precomputed:main'
    ]
}
install_requires = [
    'knossos_utils',
    'dxchange>=0.1.4',
]
setup(
    name='KLab_Utils',
    packages=['KLab_Utils'],
    entry_points=entry_points,
    include_package_data=True,
    version='1.0',
    description='Various Kasthuri Lab scripts.',
    keywords=['converter', 'skeletonization', 'segmentation'],
    install_requires=install_requires,
)
