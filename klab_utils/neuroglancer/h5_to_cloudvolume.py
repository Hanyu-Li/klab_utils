import h5py
import cloudvolume
import tqdm 
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def h5_to_cloudvolume(h5_path, cv_path, union_bbox, sub_bboxes, sub_indices, resolution, chunk_size):
  union_offset = np.array(union_bbox.minpt)
  union_size = np.array(union_bbox.maxpt) - np.array(union_bbox.minpt)
  # padded_union_size = ((union_size-1) // chunk_size + 1 ) * chunk_size
  cv_merge = prepare_precomputed(
    cv_path, 
    offset=union_offset, 
    size=union_size, 
    # size=padded_union_size, 
    resolution=resolution, 
    chunk_size=chunk_size)

  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
  local_sub_bboxes = [sub_bboxes[i] for i in sub_indices]
  with h5py.File(h5_path, 'r') as f:
    h5_ds = f['output']
    pbar = tqdm(local_sub_bboxes, desc='h5 to precomputed')
    for ffn_bb in pbar:
      abs_offset = ffn_bb.start
      abs_size = ffn_bb.size

      rel_offset = abs_offset - union_offset
      rel_size = abs_size

      # logging.warning('write %s %s', abs_offset, abs_size)
      h5_slc = np.s_[
        rel_offset[0]:rel_offset[0] + rel_size[0],
        rel_offset[1]:rel_offset[1] + rel_size[1],
        rel_offset[2]:rel_offset[2] + rel_size[2]
      ]

      cv_slc = np.s_[
        abs_offset[0]:abs_offset[0] + abs_size[0],
        abs_offset[1]:abs_offset[1] + abs_size[1],
        abs_offset[2]:abs_offset[2] + abs_size[2],
        0
      ]

      # cv_slc = [h5_slc[0], h5_slc[1], h5_slc[2], 0]
      # logging.warning('write %d %s', mpi_rank, ffn_bb)
      cv_merge[cv_slc] = h5_ds[h5_slc]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, default=None, 
    help='input_h5')
  parser.add_argument('output', type=str, default=None,
    help='output_cloudvolume')
  parser.add_argument('--resolution', type=str, default='6,6,40')
  parser.add_argument('--chunk_size', type=str, default='256,256,64')

  args = parser.parse_args()
  resolution = [int(i) for i in args.resolution.split(',')]
  chunk_size = [int(i) for i in args.chunk_size.split(',')]