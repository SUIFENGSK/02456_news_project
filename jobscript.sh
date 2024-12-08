#!/bin/sh


### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J dp-project-test
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s214919@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi

# Create job_out if it is not present
# shellcheck disable=SC3010
if [[ ! -d ${BLACKHOLE}/DeepLearning/job_out ]]; then
	mkdir "${BLACKHOLE}"/DeepLearning/job_out
fi


date=$(date +%Y%m%d_%H%M)
mkdir "${BLACKHOLE}"/DeepLearning/runs/train/"${date}"


# Activate venv
module load python3/3.10.14
module load cuda/12.2
# shellcheck disable=SC3046
source "${BLACKHOLE}"/DeepLearning/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.2.0

# run training
python3 "${BLACKHOLE}"/DeepLearning/02456_news_project/src/torch_nerd/main.py