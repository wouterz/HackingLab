#!/bin/sh

#You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=stud-ewi

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=stud-ewi

#SBATCH --account=stud-ewi-crs-in4253et

# The default run (wall-clock) time is 1 minute
#SBATCH --time=1:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=1

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=8192

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Set to use first available GPU
#SBATCH --gres=gpu

# Your job commands go below here

# Uncomment these lines when your job requires this software
module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.3.0.29
#module load matlab/R2018b
module load cuda/10.1 cudnn/10.1-7.6.0.64

# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
# For example: srun python my_program.py
# Use this simple command to check that your sbatch settings are working (verify the resources allocated in the usage statistics)
#srun hostname

srun python3 ./random_adv_script.py
