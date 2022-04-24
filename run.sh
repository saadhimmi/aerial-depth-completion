shopt -s expand_aliases
alias PYTHON="/cluster/scratch/shimmi/miniconda3/envs/torch101/bin/python"
module load eth_proxy

PYTHON -V
#pip install torch==1.0.1 torchvision==0.2.2 scipy==1.2.2 wandb
#git pull

PYTHON main.py 	--data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/dsq-montcortes-castle-45/" \
	       	--epochs 100 --num-samples 0 --workers 16 -lr 0.0001 --batch-size 32 --dcnet-arch gudepthcompnet18 \
	   	--training-mode dc1-cf1-ln1 --criterion l2
