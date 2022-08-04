#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./scripts/go.sh

num_blocks=(100 200 500 1000 2000 5000 10000 20000 50000 100000)
loss=binary_crossentropy # mean_squared_error #binary_crossentropy
snr=0

for nb in "${num_blocks[@]}"; do
    ./scripts/slurm.sh \
        -num_block $nb \
        -loss $loss \
        -train_channel_low $snr \
        -train_channel_high $snr \
        -tags data_scaling fixed_test_blocks_10000 snr_fix x_entropy
done