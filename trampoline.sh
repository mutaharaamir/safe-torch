#!/bin/bash

source activate ffcv_env 
pip install --upgrade torch
pip install -r /home/c01toja/CISPA-projects/training_dynamics-2024/SAFEgptPrune/safe-torch/requirements.txt

#pip install prettytable
#pip install terminaltables
#pip install fastargs
#pip install pandas
#pip install pyyaml
#pip install tqdm
#pip install schedulefree
#pip install airbench
#pip install pyhessian
#pip install timm
#pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
#pip install tensorboard
#pip install wandb

bash /home/c01toja/CISPA-projects/training_dynamics-2024/SAFEgptPrune/safe-torch/language/scripts/"$1" "$2" "$3" "$4"