#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpulong     # put the job into the gpu partition/queue
#SBATCH --gres=gpu:1
#SBATCH --output=log.out        # file name for stdout/stderr
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3-00:00:00       # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=parseWaymo   # job name (default is the name of this file)
#SBATCH --mail-user=pokorsi1@fel.cvut.cz    # where send info about job
#SBATCH --mail-type=ALL                     # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

module purge # unload all loaded modules

ml OpenCV/4.5.1-foss-2020b-contrib
ml TensorFlow/2.4.1-fosscuda-2020b

pip install waymo-open-dataset-tf-2-4-0

python preprocess.py /home/pokorsi1/data/waymo_flow/tfrecords/valid \
                     /home/pokorsi1/data/waymo_flow/preprocess/valid