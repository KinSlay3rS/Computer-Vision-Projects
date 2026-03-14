#!/bin/bash
#PBS -N unet_standard_ulti
#PBS -P cavity_design.spons
#PBS -l select=1:ncpus=4:ngpus=1:centos=skylake
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o out_unet_s_ultimate.log
#PBS -M ph1221236@physics.iitd.ac.in
#PBS -m abe

module load apps/apptainer/1.4.0
cd $PBS_O_WORKDIR

apptainer exec --nv --bind $PBS_O_WORKDIR:/workspace --bind /home/physics/btech/ph1221236/.cache:/home/physics/btech/ph1221236/.cache torch_full_lpips.sif /bin/bash -c "cd /workspace/src && python train.py --model unet --epochs 100 --illumination standard --tag ultimate --bilinear --attention --loss-function ultimate"





