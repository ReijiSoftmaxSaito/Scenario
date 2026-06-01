# CUDA_VISIBLE_DEVICES=1

path=/mnt/data/mvtec

method=INP_Former
# [fastflow, mambaAD, INP_Former, RDplus2,
#  UniNet, DiAD, Dinomaly, SimpleNet, GLASS, RePaste]

result=$method

categories=(
    capsule
    carpet
    grid
    hazelnut
    leather
    pill
    screw
)

scenario_type=(
    A2N
    A2N_Normal
)

case $method in
    SimpleNet)
        script="SimpleNet.py"
        ;;
    
    GLASS)
        script="GLASS.py"
        ;;
    
    RePaste)
        script="GLASS_RePaste.py"
        ;;

    *)
        script="main.py"
        ;;
esac

echo "=== Using script: $script ==="

# train
for sce in "${scenario_type[@]}"; do
for cat in "${categories[@]}"; do

    echo "=== Running method: $method ==="
    echo "=== Running category: $cat ==="
    echo "=== Running scenario: $sce ==="

    python "$script" \
        -cfg "configs/$method.yaml" \
        --data "$path" \
        --scenario "$sce" \
        -cat "$cat" \
        --result "$result"

done
done

echo "=== A2N A2N ==="

python show_data_scenario.py \
    --scenario "$method" \
    --scenario_type "A2N" \
    --output "_result_A2N_A2N.csv" \
    --mean

echo "=== A2N Normal ==="

python show_data_scenario_A2N_Normal.py \
    --method "$method" \
    --scenario "A2N_Normal" \
    --mean

# eval
# for sce in "${scenario_type[@]}"; do
# for cat in "${categories[@]}"; do

#     echo "=== Running method: $method ==="
#     echo "=== Running category: $cat ==="
#     echo "=== Running scenario: $sce ==="

#     python "$script" \
#         -cfg "configs/$method.yaml" \
#         --data "$path" \
#         --scenario "$sce" \
#         -cat "$cat" \
#         --result "$result" \
#         --eval

# done
# done