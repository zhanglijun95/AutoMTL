#!/bin/bash
#SBATCH --job-name=MTLC
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=rtx8000-long             # Partition to submit to (serial_requeue)
#SBATCH --exclude=node067
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
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

python search_cost.py --data='Cityscapes'

#python search_cost.py --data='Taskonomy'

sleep 1
exit
