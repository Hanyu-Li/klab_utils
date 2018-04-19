import sys
from setuptools import setup

entry_points = {
    'console_scripts': [
        'knossos_to_stats = KLab_Utils.knossos_to_stats:main',
        'knossos_to_tif = KLab_Utils.knossos_to_tif:main',
        'knossos_to_swc = KLab_Utils.knossos_to_swc:main',
        'swc_to_mask = KLab_Utils.swc_to_mask:main',
        'labels_to_knossos = KLab_Utils.labels_to_knossos:main',
        'fiji_stitch = KLab_Utils.fiji_stitch:main',
        'neuroglance_raw = KLab_Utils.neuroglance_raw:main',
        'neuroglance_precomputed = KLab_Utils.neuroglance_precomputed:main',
        'neuroglance_public = KLab_Utils.neuroglance_public:main',
        'raw_to_precomputed = KLab_Utils.raw_to_precomputed:main',
        'generate_mesh = KLab_Utils.generate_mesh:main',
        'generate_mesh_manifest = KLab_Utils.generate_mesh_manifest:main',
        'contrast_adjust = KLab_Utils.contrast_adjust:main',
        'offset_annotation = KLab_Utils.offset_annotation:main',
        'EM_preprocess = KLab_Utils.EM_preprocessor:main',
        'EM_trackEM2_preprocess = KLab_Utils.EM_trackEM2_preprocess:main',
        'generate_prealign_txt = KLab_Utils.generate_prealign_txt:main',
        'rename_trackEM2 = KLab_Utils.rename_trackEM2:main',
        'mesh_generator = KLab_Utils.mesh_generator:main'


    ]
}
install_requires = [
    'knossos_utils',
    'tqdm',
    'dxchange',
    'cloud-volume',
    'neuroglancer',
    'igneous'
]
setup(
    name='KLab_Utils',
    author='Hanyu Li',
    packages=['KLab_Utils'],
    entry_points=entry_points,
    include_package_data=True,
    version='1.0',
    description='Various Kasthuri Lab scripts.',
    keywords=['converter', 'skeletonization', 'segmentation'],
    install_requires=install_requires,
)
