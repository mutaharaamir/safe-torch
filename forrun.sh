#!/bin/bash

sparsity=(0.5 0.6) # 0.2 0.1 0.05)

datasets=('c4')

seeds=(42)

for sparsity in "${sparsity[@]}"; do
	for datasets in "${datasets[@]}"; do
		for seeds in "${seeds[@]}"; do
				#sbatch run.sh trampoline.sh safe.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh safe13b.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh safeLL3-8b.sh $sparsity $datasets $seeds

				sbatch run.sh trampoline.sh safeHAM.sh $sparsity $datasets $seeds

				#sbatch run.sh trampoline.sh alps.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh alps13b.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh alpsLL3-8b.sh $sparsity $datasets $seeds
				
				#sbatch run.sh trampoline.sh sparsegpt.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh sparsegpt13b.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh sparsegptLL3-8b.sh $sparsity $datasets $seeds

				#sbatch run.sh trampoline.sh wanda.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh wanda13b.sh $sparsity $datasets $seeds
				#sbatch run.sh trampoline.sh wandaLL3-8b.sh $sparsity $datasets $seeds
			done
	done
done


