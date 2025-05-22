#!/bin/bash
# Helper script to run the benchmarking

prime_upper_bounds=(
    10
    100
    1000
    10000
    100000
    1000000
    10000000
    100000000
    1000000000
)

for upper_bound in "${prime_upper_bounds[@]}"; do
    echo "Running benchmark for upper bound: $upper_bound"
    python benchmark.py "$upper_bound"
done