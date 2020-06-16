#!/bin/bash

#SBATCH -o /../../../myjob.%j.%N.out   # Output-File
#SBATCH -D /../../../                  # Working Directory
#SBATCH -J AlphaZero_train_Gruppe1	# Job Name
#SBATCH --nodes=2 		# Anzahl Knoten N
#SBATCH --ntasks-per-node=20 	# Prozesse n pro Knoten
#SBATCH --ntasks-per-core=1	# Prozesse n pro CPU-Core
#SBATCH --mem=500M              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=72:00:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vorname.nachname@tu-berlin.de

module load nvidia/cuda/10.1
module load nvidia/cudnn/7.6.5.32
module load nvidia/tensorrt/6.0.1.5

cd
source ~/env/bin/activate
cd AlphaZero
mpirun $myApplication