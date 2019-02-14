import sys
from setuptools import setup

entry_points = {
    'console_scripts': [
        'knossos_to_stats = klab_utils.knossos_to_stats:main',
        'knossos_to_tif = klab_utils.knossos_to_tif:main',
        'knossos_to_swc = klab_utils.knossos_to_swc:main',
        'swc_to_mask = klab_utils.swc_to_mask:main',
        'labels_to_knossos = klab_utils.labels_to_knossos:main',
        'fiji_stitch = klab_utils.fiji_stitch:main',
        'neuroglance_raw = klab_utils.neuroglance_raw:main',
        'neuroglance_precomputed = klab_utils.neuroglance_precomputed:main',
        'neuroglance_public = klab_utils.neuroglance_public:main',
        'raw_to_precomputed = klab_utils.raw_to_precomputed:main',
        'generate_mesh = klab_utils.generate_mesh:main',
        'generate_mesh_manifest = klab_utils.generate_mesh_manifest:main',
        'contrast_adjust = klab_utils.contrast_adjust:main',
        'offset_annotation = klab_utils.offset_annotation:main',
        'EM_preprocess = klab_utils.EM_preprocessor:main',
        'EM_trackEM2_preprocess = klab_utils.EM_trackEM2_preprocess:main',
        'generate_prealign_txt = klab_utils.generate_prealign_txt:main',
        'rename_trackEM2 = klab_utils.rename_trackEM2:main',
        'mesh_generator = klab_utils.mesh_generator:main',
        'aligntk_preprocess = klab_utils.aligntk_preprocess:main',
        'reduce_resolution = klab_utils.reduce_resolution:main'


    ]
}
install_requires = [
    'knossos_utils',
    'tqdm',
    #'dxchange',
    #'cloud-volume',
    #'neuroglancer',
    #'igneous'
]
setup(
    name='klab_utils',
    author='Hanyu Li',
    packages=['klab_utils'],
    entry_points=entry_points,
    include_package_data=True,
    version='1.0',
    description='Various Kasthuri Lab scripts.',
    keywords=['converter', 'skeletonization', 'segmentation'],
    install_requires=install_requires,
)
