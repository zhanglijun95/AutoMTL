#!/bin/bash
#SBATCH --job-name=NYUSample3
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=m40-long             # Partition to submit to (serial_requeue)
#SBATCH --exclude=node007
#SBATCH --mem=4096               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=exp/exp_output/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=exp/exp_output/run_logs_%j.err            # File to which STDERR will be written
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
__conda_setup="$('/home/lijunzhang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/home/lijunzhang/anaconda3/etc/profile.d/conda.sh" ]; then
		. "/home/lijunzhang/anaconda3/etc/profile.d/conda.sh"
	else
		export PATH="/home/lijunzhang/anaconda3/bin:$PATH"
	fi  
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
conda activate multitask
sleep 1

#python experiments_random.py --skip_random --seed=60 --data='NYUv2' --ckpt_dir='checkpoint/NYUv2/' --reload_ckpt='post_train_22800iter.model' --print_iters=150 --val_iters=300 --task_iters 50 50 50

python experiments_sample.py --seed=10 --sample_dir='sample_design3_001/' --data='NYUv2' --ckpt_dir='checkpoint/NYUv2/' --reload_ckpt='alter_train_with_reg_001_20000iter.model' --print_iters=150 --val_iters=300 --task_iters 50 50 50

sleep 1
exit
