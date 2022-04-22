#!/bin/bash
#SBATCH --job-name=CityScapes_test
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=rtx8000-long            # Partition to submit to (serial_requeue)
#SBATCH --exclude=node007
#SBATCH --mem=10240              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=exp/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=exp/run_logs_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#### SBATCH --mail-type=FAIL
#SBATCH --mail-user=yiminghuang@umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
set -x -e
##### Experiment settings #####
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/yiminghuang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yiminghuang/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yiminghuang/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yiminghuang/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate multitask
echo "activate"
sleep 1

# 
python example_mobilenetv2.py

sleep 1
exit
