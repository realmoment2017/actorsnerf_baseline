EXP=single_gpu
MODE=eval_freeview
ITER=iter_200000
GPU=7

SKIP=skip1
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/387/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/393/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/394/$EXP-$SKIP.yaml

SKIP=skip3
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/387/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/393/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/394/$EXP-$SKIP.yaml

SKIP=skip10
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/387/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/393/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/394/$EXP-$SKIP.yaml

SKIP=skip30
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/387/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/393/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/394/$EXP-$SKIP.yaml

SKIP=skip60
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/387/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/393/$EXP-$SKIP.yaml
CUDA_VISIBLE_DEVICES=$GPU python eval.py --type $MODE --cfg configs/human_nerf/zju_mocap/394/$EXP-$SKIP.yaml


#CUDA_VISIBLE_DEVICES=0,1 python vis_mesh.py --type vis_mesh --cfg configs/human_nerf/zju_mocap/tiktok/0205_mono_zju_mocap-enctrain-mweight_no-nonrigid_test-cmlp_mlp-hard_0.01-mse_0.2-lpips_1.0-test-smpl.yaml
