#!/bin/bash
#PBS -N transformer_standard_train
#PBS -P cavity_design.spons
#PBS -l select=1:ncpus=4:ngpus=1:centos=skylake
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o out_tra_st_lp.log
#PBS -M ph1221236@physics.iitd.ac.in
#PBS -m abe

module load apps/apptainer/1.4.0
cd $PBS_O_WORKDIR

apptainer exec --nv --bind $PBS_O_WORKDIR:/workspace:rw torch_full_lpips.sif /bin/bash -c "cd /workspace/src && python train.py --model transformer --epochs 50 --illumination standard --tag lpips_50 --loss-function l1_lpips"



