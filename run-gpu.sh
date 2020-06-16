#!/bin/sh

#SBATCH -J AZ_train_Gruppe1                     # Job Name

#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Tasks

#SBATCH --ntasks-per-core=2                     # Number of processes per CPU code
#SBATCH --cpus-per-task=1                       # Number of CPUs per task?
#SBATCH --mem=500G                              # Memory limit per node

#SBATCH --time=4:00:00                          # Expected maximum run time

#SBATCH --partition=gpu                         # This is needed to use a GPU
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node


#SBATCH -o out-gpu                              # send stdout to outfile
#SBATCH -e err-gpu                              # send stderr to errfile


## Job-Status per Mail
##SBATCH --mail-type=ALL
##SBATCH --mail-user={your email here}

ulimit -u 512

module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

source ~/env/bin/activate
cd ~/AlphaZero
cd azts
python3 self_play.py
deactivate
