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


# PYTHON main.py --evaluate "/cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar" \
# 			   --data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/"

# PYTHON main.py --evaluate "/cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar" \
# 			   --data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/my-sequences/"


# PYTHON main.py --output "dc PRETRAINED on loop data - 15 epochs - batch 32 - lr 1e-4" \
# 		    --data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/my-sequences/" \
# 			--epochs 15 --num-samples 0 --workers 16 --criterion l2 -lr 0.0001 --batch-size 32 --training-mode dc1_only --dcnet-arch ged_depthcompnet \
# 			--dcnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights

# mkdir $TMPDIR/data
# tar -xf /cluster/work/riner/users/PLR-2021/dont-share-my-face/irchel140821stars_0_and_10.tar -C $TMPDIR/data

# PYTHON main.py  --output "IRCHEL (with max depth - full data) - PRETRAINED end-to-end - L1 criterion - 30 epochs - batch 32 - lr 1e-4" \
# 		--data-path $SCRATCH/aerial-depth-completion/full-ircheldata/sequences/train --max-gt-depth 399.75 \
# 		--epochs 30 --num-samples 0 --workers 16 --criterion l1 -lr 0.0001 --batch-size 32 --training-mode dc1-cf1-ln1 \
# 		--dcnet-arch ged_depthcompnet --dcnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights \
# 		--lossnet-arch ged_depthcompnet --lossnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:lossdc_weights


# PYTHON main.py --output "IRCHEL (with max depth) - dc PRETRAINED dc1_only - l2 criterion - 30 epochs - batch 32 - lr 1e-4" \
# 		    --data-path $SCRATCH/tmp_data --max-gt-depth 399.75 \
# 			--epochs 30 --num-samples 0 --workers 16 --criterion l2 -lr 0.0001 --batch-size 32 --training-mode dc1_only --dcnet-arch ged_depthcompnet \
# 			--dcnet-pretrained /cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights

# PYTHON main.py --evaluate "/cluster/scratch/shimmi/aerial-depth-completion/data/checkpoints/aerial-only-visim/model_best.pth.tar" \
# 			   --data-path "/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/aerial-only/"

# train_arr=(xxx h0p30 h0p45 h0p60 h0p75 h0p90 h10p30 h10p45 h10p60 h10p75 h10p90)
# for i in {1..10}
# do
#	PYTHON main.py --output "new_eval_${train_arr[$i]}" \
#				--evaluate "/cluster/home/shimmi/aerial-depth-completion/best-l2.pth.tar" \
#				--num-samples 0 --max-gt-depth 399.75 \
#				--data-path "$SCRATCH/aerial-depth-completion/full-ircheldata/sequences/val/t$i"
#
#	tar -cvf /cluster/scratch/shimmi/transformed-train/transformed_${train_arr[$i]}.tar /cluster/scratch/shimmi/aerial-depth-completion/results/new_eval_${train_arr[$i]}/transformed
#done

eval_arr=(xxx h20p30 h20p45 h20p60 h20p75 h20p90 h30p30 h30p45 h30p60 h30p75 h30p90 h40p60 h40p75 h50p75 h50p90 h80p75 h80p90)
for i in {16..16}
do
	PYTHON main.py --output "new_eval_${eval_arr[$i]}" \
				--evaluate "/cluster/home/shimmi/aerial-depth-completion/best-l2.pth.tar" \
				--num-samples 0 --max-gt-depth 399.75 \
				--data-path "$SCRATCH/aerial-depth-completion/full-ircheldata/sequences/val/$i"

	tar -cvf /cluster/scratch/shimmi/transformed-data/transformed_${eval_arr[$i]}.tar /cluster/scratch/shimmi/aerial-depth-completion/results/new_eval_${eval_arr[$i]}/transformed
done

# train_arr=(h0p30 h0p45 h0p60 h0p75 h0p90 h10p30 h10p45 h10p60 h10p75 h10p90)
# PYTHON main.py --output "debug_eval_${train_arr[0]}" \
# 			--evaluate "/cluster/home/shimmi/aerial-depth-completion/best-l2.pth.tar" \
# 			--num-samples 0 --max-gt-depth 399.75 \
# 			--data-path "$SCRATCH/aerial-depth-completion/full-ircheldata/sequences/val/1"
