#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --time=01:00:00
#SBATCH --mem=4G

#creating venv here. :')
module load anaconda/2023.07.tuftsai

conda create -n myenv python=3.8
conda activate myenv
conda install scipy numpy scikit-learn pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pillow matplotlib

python simple_encoder.py

