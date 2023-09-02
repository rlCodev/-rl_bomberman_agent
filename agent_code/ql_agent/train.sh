#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

# Assign the first argument to the variable 'n'
n=$1

# Loop 'n' times and call the Python script
for ((i=1; i<=$n; i++)); do
    echo "Running iteration $i"
    python ../../main.py play --agents ql_agent --train 1 --no-gui
done

# Call this to train model: chmod +x traub.sh
# ./train.sh 100