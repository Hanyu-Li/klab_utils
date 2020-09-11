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
        'klab_utils.knossos.stack_to_wkcube = klab_utils.knossos.stack_to_wkcube:main',
        'klab_utils.knossos.wkw_io = klab_utils.knossos.wkw_io:main',
        'klab_utils.knossos.cube_wkw = klab_utils.knossos.cube_wkw:main',
        'klab_utils.knossos.wkw_to_h5 = klab_utils.knossos.wkw_to_h5:main',


        # trakem2 related
        'klab_utils.trakem2.preprocess_stack = klab_utils.trakem2.preprocess_stack:main',
        'klab_utils.trakem2.preprocess_tiles = klab_utils.trakem2.preprocess_tiles:main',
        'klab_utils.trakem2.preprocess_tiles_thermo = klab_utils.trakem2.preprocess_tiles_thermo:main',
        'klab_utils.trakem2.generate_prealign_txt = klab_utils.trakem2.generate_prealign_txt:main',
        'klab_utils.trakem2.rename = klab_utils.trakem2.rename:main',
        'klab_utils.trakem2.mpi_montage = klab_utils.trakem2.mpi_montage:main',
        'klab_utils.trakem2.mpi_export = klab_utils.trakem2.mpi_export:main',
        'klab_utils.trakem2.align = klab_utils.trakem2.align:main',

        # neuroglancer related
        'klab_utils.neuroglancer.neuroglance_precomputed = klab_utils.neuroglancer.neuroglance_precomputed:main',
        'klab_utils.neuroglancer.raw_to_precomputed = klab_utils.neuroglancer.raw_to_precomputed:main',
        'klab_utils.neuroglancer.mesh_generator = klab_utils.neuroglancer.mesh_generator:main',
        'klab_utils.neuroglancer.mesh_manifest_generator = klab_utils.neuroglancer.mesh_manifest_generator:main',
        'klab_utils.neuroglancer.skeleton_generator = klab_utils.neuroglancer.skeleton_generator:main',
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
        'klab_utils.aligntk.clahe = klab_utils.aligntk.clahe:main',
        'klab_utils.aligntk.mpi_apply_map = klab_utils.aligntk.mpi_apply_map:main',
        'klab_utils.aligntk.fix_defects = klab_utils.aligntk.fix_defects:main',
        'klab_utils.aligntk.h5_to_stack = klab_utils.aligntk.h5_to_stack:main',
        'klab_utils.aligntk.correct_artifact = klab_utils.aligntk.correct_artifact:main',

        # ffn 
        'klab_utils.ffn.write_h5 = klab_utils.ffn.write_h5:main',
        'klab_utils.ffn.downsample_h5 = klab_utils.ffn.downsample_h5:main',
        'klab_utils.ffn.export_inference = klab_utils.ffn.export_inference:main',
        'klab_utils.ffn.agglomerate = klab_utils.ffn.agglomerate:main',
        'klab_utils.ffn.mpi_agglomerate = klab_utils.ffn.mpi_agglomerate:main',
        'klab_utils.ffn.mpi_agglomerate_v2 = klab_utils.ffn.mpi_agglomerate_v2:main',
        'klab_utils.ffn.mpi_agglomerate_h5 = klab_utils.ffn.mpi_agglomerate_h5:main',
        'klab_utils.ffn.reconciliate = klab_utils.ffn.reconciliate:main',
        'klab_utils.ffn.reconciliate_remap = klab_utils.ffn.reconciliate_remap:main',
        'klab_utils.ffn.reconciliate_find_graph = klab_utils.ffn.reconciliate_find_graph:main',
        'klab_utils.ffn.reconciliate_agglomerate = klab_utils.ffn.reconciliate_agglomerate:main',
        'klab_utils.ffn.inspect_precoms = klab_utils.ffn.inspect_precoms:main',
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
