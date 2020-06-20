#!/bin/bash

#SBATCH -J AlphaZero_train_Gruppe1	# Job Name

#SBATCH --nodes=9 		# Anzahl Knoten N
#SBATCH --ntasks-per-node=1 	# Prozesse n pro Knoten
#SBATCH --ntasks-per-core=1	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=01:10:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node
 

#SBATCH -o MM/logfile_MM                  # send stdout to outfile
#SBATCH -e MM/errfile_MM                  # send stderr to errfile

module load comp/gcc/7.2.0 
module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

cd
source ~/env/bin/activate

ssh -f -N  mlflow_user@35.223.113.101 -L 0.0.0.0:5050:35.223.113.101:8000 -o TCPKeepAlive=yes
ssh -f -N  mlflow_user@35.223.113.101 -L 0.0.0.0:5054:35.223.113.101:22 -o TCPKeepAlive=yes
chmod a+x /home/users/t/t.moussa/AlphaZero/Interpreter/Engine/stockfish-x86_64


export GIT_PYTHON_REFRESH=quiet
export MLFLOW_ARTIFACT_URI='sftp://mlflow/home/mlflow_user/mlflow/mlruns/55'
export MLFLOW_TRACKING_URI='http://frontend02:5050'


mlflow run AlphaZero -e train --no-conda --experiment-id=55 
 
deactivate
