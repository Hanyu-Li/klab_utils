#!/bin/bash
PROC=8
IMAGE_DIR="../stack/"
GROUP_SIZE=$(($PROC-1))
N_IMAGES=`ls $IMAGE_DIR | wc -l`
N_GROUPS=$(($N_IMAGES / $GROUP_SIZE))
GROUPS=`seq 2 $(($N_GROUPS - 1))`
echo "Total image count: $N_IMAGES"
echo "Number of groups: $N_GROUPS"

mkdir -p logs
mkdir -p masks
mkdir -p cmaps
mkdir -p logs
mkdir -p amaps
mkdir -p grids
mkdir -p maps
mkdir -p aligned

echo "10   1.0  0.1
 9   1.0  0.1
 8   1.0  0.3
 7   1.0  1.0
 7   1.0  2.0
 7   1.0  5.0
 6   1.0  5.0" > schedule.lst

# aligntk_cut_to_eight --input_dir ./images_old --output_dir ./images
aligntk_preprocess --image_dir $IMAGE_DIR --output_dir . --group_size $GROUP_SIZE
aligntk_gen_mask --image_dir $IMAGE_DIR --mask_dir ./masks

for i in $GROUPS;
do
  echo "find_rst: $i"
  mpirun -np $PROC find_rst -pairs pairs$i.lst -tif -images $IMAGE_DIR -mask masks/ -output cmaps/ -max_res 1024 -scale 1.0 -tx -50-50 -ty -50-50 -summary cmaps/summary$i.out
done | tqdm --total $N_GROUPS


for i in $GROUPS;
do
  echo "register: $i"
  mpirun -np $PROC register -pairs pairs$i.lst -images $IMAGE_DIR -mask masks/ -tif -output maps/ -distortion 6.0 -output_level 6 -depth 6 -quality 0.5 -summary maps/summary$i.out -initial_map cmaps/
done | tqdm --total $N_IMAGES

echo 'align'
FIXED=`head images.lst -n 1`
mpirun -np $PROC align -images $IMAGE_DIR -image_list images.lst -map_list pairs.lst -maps maps/ -output amaps/ -schedule schedule.lst -incremental -output_grid grids/ -grid_size 1024x1024 -fixed $FIXED -fold_recovery 10

echo "apply_map"
apply_map -image_list images.lst -images $IMAGE_DIR -maps amaps/ -output aligned/ -memory 50000
