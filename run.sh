for seed in 12 17 22
do
    echo "Seed:" ${seed}
    path=results/longformer_jit_cm_bs64lr1e-5_${seed}
    mkdir -p ${path}
    python3 train.py --seed ${seed} --path "${path}/model.weights"
    python3 eval.py --seed ${seed} --path "${path}/model.weights"
done