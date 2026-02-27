# 运行 ubuntu20.04-base 镜像，挂载当前 GLIP 目录
docker run -it  --gpus all --name ubuntu-glip  -v $(pwd):/workspace/GLIP  -w /workspace/GLIP   ubuntu20.04-base:latest



python -m torch.distributed.launch --nproc_per_node=2  tools/train_net.py   --config-file configs/pretrain/labelme_grounding.yaml     --skip-test


python -m torch.distributed.launch --nproc_per_node=2  tools/train_net.py    --config-file configs/pretrain/glip_Swin_T_O365_GoldG_my_model.yaml     --skip-test   --use-tensorboard    --override_output_dir    output/coco_dataset0227_glip_tiny_model_o365_goldg_cc_sbu   SOLVER.CHECKPOINT_PERIOD 500 

