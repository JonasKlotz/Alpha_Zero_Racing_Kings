#!/bin/sh
sbatch /home/users/k/konstantin_ausborn/AlphaZero-Gruppe1/start_skript.sh
while :
do
    squeue -l -u konstantin_ausborn;
#   sleep 5; 
    read -t 0.3 -n 1 k
    [[ "$k" == 's' ]] && break
done 
clear

