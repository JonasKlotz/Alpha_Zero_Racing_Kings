#!/bin/sh

#SBATCH -J AlphaZero_train_Gruppe1              # Job Name

## Leave these values as they are unless you know what you are doing
#SBATCH --ntasks=1                              # Tasks
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-core=2                     # Number of processes per CPU code
#SBATCH --cpus-per-task=1                       # Number of CPUs per task?
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node
#SBATCH --partition=gpu                         # This is needed to use a GPU


## Adjust to your needs
#SBATCH --mem=25G                               # Memory limit per node
#SBATCH --time=0:01:00                          # Expected maximum run time

##SBATCH -c 4                                   # cores requested
#SBATCH -o outfile                              # send stdout to outfile
#SBATCH -e errfile                              # send stderr to errfile
#SBATCH -t 0:01:00                              # time requested in hour:minute:second


## Job-Status per Mail
##SBATCH --mail-type=ALL
##SBATCH --mail-user={your email here}

ulimit -u 512
# benötigte SW / Bibliotheken laden (CUDA, etc.)
module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

##STORAGE_DEFAULT_DIRECTORY="$PWD/storage/"

#commands to be executed
cd
source ~/env/bin/activate
cd AlphaZero
python3 test_run.py
deactivate

##scontrol show job $SLURM_JOBID                # for debugin
