# CUDA_VISIBLE_DEVICES=1

path=/mnt/HDD1/dataset/MVtec_AD_N2A_NEW/

method=patchcore 
# [fastflow, mambaAD, INP_Former, RDplus2,
#  UniNet, DiAD, Dinomaly, SimpleNet, GLASS, RePaste]

result=$method

categories=(
    bottle
    cable
    capsule
    carpet
    grid
    hazelnut
    leather
    metal_nut
    pill
    screw
    tile
    toothbrush
    transistor
    wood
    zipper
)
scenario_type=(
    N2A
    N2A_Normal
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


for sce in "${scenario_type[@]}"; do
echo "=== N2A ==="
python show_data_scenario.py \
    --scenario "$method" \
    --scenario_type "$sce" \
    --output "_result_$sce.csv" \
    --mean
done

