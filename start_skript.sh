#!/bin/bash

#SBATCH -J AlphaZero_Training

#SBATCH --nodes=1 			# Anzahl Knoten
#SBATCH --ntasks-per-node=1 		# Prozesse n pro Knoten
#SBATCH --ntasks-per-core=1	  	# Prozesse n pro CPU-Core
#SBATCH --mem=5G              		# resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=02:00:00 		# Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1              # Use 1 GPU per node

#SBATCH -o /home/users/k/konstantin_ausborn/AlphaZero-Gruppe1/stdout.log
#SBATCH -e /home/users/k/konstantin_ausborn/AlphaZero-Gruppe1/stderr.log

module load comp/gcc/7.2.0 
module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

cd
cd AlphaZero-Gruppe1
source venv/bin/activate

#chmod a+x /home/users/t/t.moussa/AlphaZero/Interpreter/Engine/stockfish-x86_64

export GIT_PYTHON_REFRESH=quiet
export MLFLOW_ARTIFACT_URI='sftp://mlflow/home/mlflow_user/mlflow/mlruns/57'
export MLFLOW_TRACKING_URI='http://frontend02:5050'

mlflow run ../AlphaZero-Gruppe1 -e train -P max_games=3000 -P player=Player/ShoutingDonkey.yaml -P max_iter=1 -P max_epochs=80 --no-conda --experiment-id=58
 
deactivate
