#!/bin/bash

# RANDOM_SEEDS=(2 17 23 42 56)
RANDOM_SEEDS=(42)
for SEED in "${RANDOM_SEEDS[@]}"; do 
    echo "Running training with Seed $SEED"
    sh scripts/train.sh -d cstdataset -g 1 -c combined_config_features_full -n random_seed_${SEED} -s $SEED
done
