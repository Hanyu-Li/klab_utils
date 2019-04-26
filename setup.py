import sys
from setuptools import setup

entry_points = {
    'console_scripts': [
        # knossos related
        'klab_utils.knossos.knossos_to_stats = klab_utils.knossos.knossos_to_stats:main',
        'klab_utils.knossos.knossos_to_tif = klab_utils.knossos.knossos_to_tif:main',
        'klab_utils.knossos.knossos_to_swc = klab_utils.knossos.knossos_to_swc:main',
        'klab_utils.knossos.swc_to_mask = klab_utils.knossos.swc_to_mask:main',
        'klab_utils.knossos.labels_to_knossos = klab_utils.knossos.labels_to_knossos:main',
        'klab_utils.knossos.offset_annotation = klab_utils.knossos.offset_annotation:main',

        # trakem2 related
        'klab_utils.trakem2.preprocess_stack = klab_utils.trakem2.preprocess_stack:main',
        'klab_utils.trakem2.preprocess_tiles = klab_utils.trakem2.preprocess_tiles:main',
        'klab_utils.trakem2.generate_prealign_txt = klab_utils.trakem2.generate_prealign_txt:main',
        'klab_utils.trakem2.rename = klab_utils.trakem2.rename:main',
        'klab_utils.trakem2.mpi_montage = klab_utils.trakem2.mpi_montage:main',
        'klab_utils.trakem2.mpi_align = klab_utils.trakem2.mpi_align:main',

        # neuroglancer related
        'klab_utils.neuroglancer.neuroglance_precomputed = klab_utils.neuroglancer.neuroglance_precomputed:main',
        'klab_utils.neuroglancer.raw_to_precomputed = klab_utils.neuroglancer.raw_to_precomputed:main',
        'klab_utils.neuroglancer.raw_to_precomputed_v2 = klab_utils.neuroglancer.raw_to_precomputed_v2:main',
        'klab_utils.neuroglancer.mesh_generator = klab_utils.neuroglancer.mesh_generator:main',
        'klab_utils.neuroglancer.add_mip_level = klab_utils.neuroglancer.add_mip_level:main',

        # aligntk_related
        'klab_utils.aligntk.contrast_adjust = klab_utils.aligntk.contrast_adjust:main',
        'klab_utils.aligntk.preprocess = klab_utils.aligntk.preprocess:main',
        'klab_utils.aligntk.cut_to_eight = klab_utils.aligntk.cut_to_eight:main',
        'klab_utils.aligntk.gen_mask = klab_utils.aligntk.gen_mask:main',
        'klab_utils.aligntk.merge_masks = klab_utils.aligntk.merge_masks:main',
        'klab_utils.aligntk.cut = klab_utils.aligntk.cut:main',
        'klab_utils.aligntk.preview = klab_utils.aligntk.preview:main',
        'klab_utils.aligntk.reduce = klab_utils.aligntk.reduce:main',
        'klab_utils.aligntk.min_rect = klab_utils.aligntk.min_rect:main',
        'klab_utils.aligntk.invert = klab_utils.aligntk.invert:main',
        'klab_utils.aligntk.invert_bg = klab_utils.aligntk.invert_bg:main',
        'klab_utils.aligntk.mpi_apply_map = klab_utils.aligntk.mpi_apply_map:main',
        'klab_utils.aligntk.fix_defects = klab_utils.aligntk.fix_defects:main',
    ]
}
install_requires = [
    'tqdm',
    'opencv-python',
    'mpi4py',
    'scikit-image',
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
