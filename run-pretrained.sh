shopt -s expand_aliases
alias PYTHON="/cluster/scratch/shimmi/miniconda3/envs/torch101/bin/python"
module load eth_proxy

PYTHON -V
#pip install torch==1.0.1 torchvision==0.2.2 scipy==1.2.2 wandb
#git pull

# PYTHON main.py  --output "PRETRAINED end-to-end with frozen unguided - 30 epochs - batch 32 - lr 1e-4" \
# 		--data-path  "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/dsq-montcortes-castle-45/" \
# 		--epochs 30 --num-samples 0 --workers 16 --criterion l2 -lr 0.0001 --batch-size 32 --training-mode dc1-cf1-ln1 \
# 		--dcnet-arch ged_depthcompnet --dcnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights \
# 		--lossnet-arch ged_depthcompnet --lossnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:lossdc_weights

# PYTHON main.py  --output "dc PRETRAINED with frozen unguided - 30 epochs - batch 32 - lr 1e-4" \
# 		--data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/dsq-montcortes-castle-45/" \
# 		--epochs 30 --num-samples 0 --workers 16 --criterion l2 -lr 0.0001 --batch-size 32 --training-mode dc1_only \
# 		--dcnet-arch ged_depthcompnet --dcnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights



PYTHON main.py --evaluate "/cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial_only_visim/model_best.pth.tar" \
			   --data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/dsq-montcortes-castle-45/"