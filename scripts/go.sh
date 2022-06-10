#!/bin/bash
# Highest level experiment script.
# This script should be launched from the root of the repository, i.e.
# ./scripts/go.sh

num_blocks=(100 500 1000 5000 10000 50000)
for nb in "${num_blocks[@]}"; do
    ./scripts/slurm.sh -num_block $nb
done