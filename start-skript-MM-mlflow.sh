#!/bin/bash

#SBATCH -J AlphaZero_train_Gruppe1	# Job Name

#SBATCH --nodes=8 		# Anzahl Knoten N
#SBATCH --ntasks-per-node=1 	# Prozesse n pro Knoten
#SBATCH --ntasks-per-core=1	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:05:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard

#SBATCH -o MM/outfile_MM                  # send stdout to outfile
#SBATCH -e MM/errfile_MM                  # send stderr to errfile

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tarik.moussa@tu-berlin.de



module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

cd
source ~/env/bin/activate

mlflow run AlphaZero -e watch_and_learn --no-conda -experiment-id=55
deactivate
