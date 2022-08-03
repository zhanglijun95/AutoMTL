#!/bin/bash
#SBATCH --job-name=PTH
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gypsum-2080ti             # Partition to submit to (serial_requeue)
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=../exp/paper/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=../exp/paper/run_logs_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lijunzhang@cs.umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
set -x -e
##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/apps/miniconda/4.8.3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/modules/apps/miniconda/4.8.3/etc/profile.d/conda.sh" ]; then
		. "/modules/apps/miniconda/4.8.3/etc/profile.d/conda.sh"
	else
		export PATH="/modules/apps/miniconda/4.8.3/bin:$PATH"
	fi  
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
conda activate multitask
sleep 1

#python ablation.py --save_dir='no_pre' #NP
#python ablation.py --pre_train --save_dir='pre_re' #PR
#python ablation.py --pre_train --fine_tune --save_dir='pre_fine' #PF
#python ablation.py --pre_train --direct_retrain --save_dir='pre_fine' #DR

#python ablation.py --pre_train --save_dir='pre_re' --policy_lambda=0.01 #PR01
#python ablation.py --pre_train --save_dir='pre_re' --policy_lambda=0.001 #PR001
#python ablation.py --pre_train --save_dir='pre_re' --policy_lambda=0.0001 #PR0001

#python ablation.py --pre_train --save_dir='pre_re_sh' --policy_lambda=0.01 --shared=30 #PR30
#python ablation.py --pre_train --save_dir='pre_re_sh' --policy_lambda=0.001 --shared=25 #PR25
#python ablation.py --pre_train --save_dir='pre_re_sh' --policy_lambda=0.0005 --shared=20 #PR20
#python ablation.py --pre_train --save_dir='pre_re_sh' --policy_lambda=0.0001 --shared=10 #PR10

python pt_hour.py --data='NYUv2'

sleep 1
exit