EXP=single_gpu
MODE=eval_freeview
ITER=iter_200000
GPU=7

SKIP=skip1
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d16/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d17/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d18/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d19/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d20/$EXP-$SKIP.yaml

SKIP=skip3
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d16/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d17/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d18/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d19/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d20/$EXP-$SKIP.yaml

SKIP=skip10
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d16/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d17/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d18/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d19/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d20/$EXP-$SKIP.yaml

SKIP=skip30
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d16/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d17/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d18/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d19/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d20/$EXP-$SKIP.yaml

SKIP=skip60
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d16/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d17/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d18/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d19/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/AIST_mocap/d20/$EXP-$SKIP.yaml


