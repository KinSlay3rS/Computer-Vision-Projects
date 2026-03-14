#!/bin/bash
#PBS -N unrolled_vortex
#PBS -P cavity_design.spons
#PBS -l select=1:ncpus=4:ngpus=1:centos=skylake
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o unrolled_vortex.log
#PBS -M ph1221236@physics.iitd.ac.in
#PBS -m abe

module load apps/apptainer/1.4.0
cd $PBS_O_WORKDIR

apptainer exec --nv --bind $PBS_O_WORKDIR:/workspace:rw torch_full_lpips.sif /bin/bash -c "cd /workspace/src && python train_unrolled.py --epochs 50 --illumination vortex --batch-size 16 --steps 12"






