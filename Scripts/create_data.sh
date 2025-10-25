#!/bin/bash

# Script to run the commands sequentially for milestones 1 to 8

for i in {16..25}; do
    echo "Running command for milestone $i"
    python main.py --name jumping --config_file ./Config/jumping.yaml --gpu 0 --sample 0 --milestone $i
done