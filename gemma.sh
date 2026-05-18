#!/bin/bash

for run in $(seq 2 10); do
  python prompt_infl_exps_medgemma.py --model medgemma-1.5-4b-it --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run $run --savedir /mnt/d/Naved/Outputs/prompt_influence/medgemma-1.5-4b-it-orig --prompt_type not_concerned
done

for run in $(seq 1 10); do
  python prompt_infl_exps_medgemma.py --model medgemma-1.5-4b-it --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run $run --savedir /mnt/d/Naved/Outputs/prompt_influence/medgemma-1.5-4b-it-orig --prompt_type neutral
done

for run in $(seq 1 10); do
  python prompt_infl_exps_medgemma.py --model medgemma-1.5-4b-it --data_dir /mnt/d/Naved/Data/ISIC99/images/originals --run $run --savedir /mnt/d/Naved/Outputs/prompt_influence/medgemma-1.5-4b-it-orig --prompt_type concerned
done