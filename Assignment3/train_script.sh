#!/bin/bash
#PBS -m bea
#PBS -M tobias.fink42@gmail.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)
python3 -u /dlvc/assignments/assignment3/group4/train_best_model.py