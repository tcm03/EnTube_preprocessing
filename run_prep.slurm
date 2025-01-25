#!/bin/bash
#SBATCH --job-name=imptask
#SBATCH --partition=batch
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160GB
#SBATCH --time=24:00:00
#SBATCH --output=/media02/nthuy/data/EnTube_preprocessing/logs/%j.out
#SBATCH --error=/media02/nthuy/data/EnTube_preprocessing/logs/%j.err

cd /media02/nthuy/data/EnTube_preprocessing
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_data
python preprocessing/main.py --folders /media02/nthuy/data/entube/EnTube/data/0 /media02/nthuy/data/entube/EnTube/data/1 /media02/nthuy/data/entube/EnTube/data/2 --config_file preprocessing/config.json
