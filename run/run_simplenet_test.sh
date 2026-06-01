#!/bin/bash

# CUDA_VISIBLE_DEVICES=1
path_A2N=/mnt/HDD1/dataset/MVtec_AD
path_N2A=/mnt/HDD1/dataset/MVtec_AD_N2A_NEW/
method=SimpleNet
result=$method

categories_A2N=(
    capsule carpet grid hazelnut leather pill screw
)

categories_N2A=(
    bottle cable capsule carpet grid hazelnut leather 
    metal_nut pill screw tile toothbrush transistor wood zipper
)

scenario_type=("A2N" "N2A")

for sce in "${scenario_type[@]}"; do
    if [ "$sce" == "A2N" ]; then
        current_categories=("${categories_A2N[@]}")
        path=$path_A2N
    else
        current_categories=("${categories_N2A[@]}")
        path=$path_N2A
    fi

    echo "======= Starting Scenario: $sce ======="

    for cat in "${current_categories[@]}"; do
        echo "=== Running method: $method | category: $cat | scenario: $sce ==="

        python SimpleNet_test.py \
            -cfg configs/$method.yaml \
            --data "$path" \
            --scenario "$sce" \
            -cat "$cat" \
            --result "$result"
    done

    echo "=== Final Aggregation for $sce ==="
    python test_result.py \
        --root "/mnt/saito/Scenario/result/$result/$sce" \
        --scenario "$sce"
done