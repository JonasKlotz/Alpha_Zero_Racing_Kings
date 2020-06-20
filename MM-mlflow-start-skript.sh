#!/bin/bash

#SBATCH -J AlphaZero_train_Gruppe1	# Job Name

#SBATCH --nodes=8 		# Anzahl Knoten N
#SBATCH --ntasks-per-node=1 	# Prozesse n pro Knoten
#SBATCH --ntasks-per-core=1	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:40:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node
 

#SBATCH -o MM/outfile_MM                  # send stdout to outfile
#SBATCH -e MM/errfile_MM                  # send stderr to errfile

module load comp/gcc/7.2.0 
module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

cd
source ~/env/bin/activate

export GIT_PYTHON_REFRESH=quiet
export MLFLOW_ARTIFACT_URI='sftp://mlflow/home/mlflow_user/mlflow/mlruns/55'
export MLFLOW_TRACKING_URI='http://frontend02:5050'
chmod a+x /home/users/t/t.moussa/AlphaZero/Interpreter/Engine/stockfish-x86_64
mlflow run AlphaZero -e create_dataset --no-conda --experiment-id=55 
 
deactivate
