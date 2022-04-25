# Image-based Place Recognition using a Depth Prediction Network

## Aerial Depth Prediction Network - Saad

In this first section, I will specify the small subtleties related to my semester project. For more general instructions, refer to the second section below.

Overall, the main two changes are: 
- Introduction of Weights and Biases (WandB, W&B) for easier experiment, hyperparameter, loss and metrics monitoring. All the experiments logs carried for the semester project are still available [on this website](https://wandb.ai/saadhimmi/semester-project-DPN)
- Introduction of my own dataloader for images coming from VI-sensor simulator following my own data organization (```VISIMDataset```). Main advantage is the ability to consider all the data (all the sequences) as a single dataset and randomly split between them to get training and validation split. Note: this feature is enabled with ```FLAG_onedataset``` and the split ratio is set by ```train_data_ratio```, both in [main.py](https://github.com/saadhimmi/aerial-depth-completion/blob/e008ecf2f62bf4c77b22baad0098a3dcc0173d13/main.py).

To get a working depth estimation network, you have to set ```--num-samples 0``` and ```--max-gt-depth 399.75``` as the maximum depth from VISIM is 400 meters.
Make sure to install the requirements using ```pip``` in a virtual environment following the General instructions (further below).

### Data
For my semester project, data has been generated on VI-sensor simulator, Irchel 3D model via AWS instances. In particular, I recorded Star trajectories at specific height (h) and pitch (p) pairs (c.f. Final Report).

In the filesystem, data is organized this way:
```
path/to/sequences/train/or/val
	h20p90
	    color
		timestamp1.png
		timestamp2.png
	    depth
		timestamp1.exr
		timestamp2.exr
		...
	h__p__
	     color
	         ...
	     depth
	         ...
	...
```
To easily reproduce core functionalities of this repository, you can download two sequences [here](https://www.sendbig.com/view-files?Id=0b0adf8e-97df-5098-f025-ec6ddbfcef2e): h20p90 (used in training) and h30p75 (for evaluation).

### Training
Best training results have been obtained with the following command:
```
python main.py --output "IRCHEL (with max depth - full data) - PRETRAINED end-to-end - L2 criterion - 30 epochs - batch 32 - lr 1e-4" \ 
		--data-path /path/to/your/sequences/train \
		--max-gt-depth 399.75 --epochs 30 --num-samples 0 --workers 16 --criterion l2 -lr 0.0001 --batch-size 32 \
		--training-mode dc1-cf1-ln1 --dcnet-arch ged_depthcompnet --lossnet-arch ged_depthcompnet \
		--dcnet-pretrained /path/to/depth/completion/checkpoints/aerial-only-visim/model_best.pth.tar:dc_weights \
		--lossnet-pretrained /path/to/depth/completion/checkpoints/aerial-only-visim/model_best.pth.tar:lossdc_weights
```
You can download the best model weights for this semester project [here](https://drive.google.com/file/d/1BcxxDLj1mmIAMl9AcyqfJT3KVm4gd7yo/view?usp=sharing) or use the checkpoint in the repo [(here)](https://github.com/saadhimmi/aerial-depth-completion/blob/4cc7c91d2b4c8d34130265b7ccdcc71d2a5dcce3/weights/best_weights.pth.tar). For the best depth completion checkpoint (used above for transfer learning), they can be found [here](https://drive.google.com/drive/folders/1D6HYo5OX0V2PAO1m2YdTsPQbGiizbhgj?usp=sharing), as described in the General Instructions.

### Evaluation
One can run evaluation on a single sequence (here height-offset 30 and pitch 60) using the following command:
```
python main.py --output "your_run_name" \
		--evaluate "/path/to/best.pth.tar" \
		--num-samples 0 --max-gt-depth 399.75 \
		--data-path "path/to/ircheldata/sequences/val/h30p60"
```
Make sure to fill the right path to your model weights and to the specific sequence you want to evaluate on.



## Aerial Depth Completion - General instructions

This work is described in the letter "Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation", by Lucas Teixeira, Martin R.
Oswald, Marc Pollefeys, Margarita Chli, published in the IEEE
Robotics and Automation Letters (RA-L / ICRA) [ETHZ Library link](https://doi.org/10.3929/ethz-b-000392181).

### Video:
<a href="https://www.youtube.com/embed/IzfFNlYCFHM" target="_blank"><img src="http://img.youtube.com/vi/IzfFNlYCFHM/0.jpg" 
alt="Mesh" width="240" height="180" border="10" /></a>

### Presentation:
<a href="https://www.youtube.com/embed/k2WH1WlYHKc" target="_blank"><img src="http://img.youtube.com/vi/k2WH1WlYHKc/0.jpg" 
alt="Mesh" width="240" height="180" border="10" /></a>

### Citations:
If you use this Code or Aerial Dataset, please cite the following publication:

```
@article{Teixeira:etal:RAL2020,
    title   = {{Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation}},
    author  = {Lucas Teixeira and Martin R. Oswald and Marc Pollefeys and Margarita Chli},
    journal = {{IEEE} Robotics and Automation Letters ({RA-L})},
    doi     = {10.1109/LRA.2020.2967296},
    year    = {2020}
}
```
NYUv2, CAB and PVS datasets require further citation from their authors. 
During our research, we reformat and created ground-truth depth for the CAB and PVS datasets. 
This code also contains thirt-party networks used for comparison. 
Please also cite their authors properly in case of use. 


#### Acknowledgment:
The authors thank [Fangchang Ma](https://github.com/fangchangma) and [Abdelrahman Eldesokey](https://github.com/abdo-eldesokey) for sharing their code that is partially used here. The authors also thanks the owner of the 3D models used to build the dataset. They are identified in each 3D model file.

### Data and Simulator

#### Trained Models

Several trained models are available - [here](https://drive.google.com/drive/folders/1D6HYo5OX0V2PAO1m2YdTsPQbGiizbhgj?usp=sharing).

#### Datasets
* Aerial Dataset - [link](https://zenodo.org/record/3614761)
* NYUv2 Dataset - [link](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz) (preprocessed by Fangchang Ma and originally from [Silberman et al. ECCV12](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html))
* CAB Dataset - on request (In this work, we created the depth information for the dataset originally published in [Teixeira and Chli IROS16](https://github.com/VIS4ROB-lab/mesh_based_mapping))
* PVS Dataset - on request (In this work, we created the depth information for the dataset originally published in [Restrepo et al. P&RS14](https://www.sciencedirect.com/science/article/pii/S0924271614002354))

To be used together by our code, the datasets need to be merged, this means that the content of the **train** folder of each dataset need to be place in a single **train** folder. The same happens with the **eval** folder.

#### Simulator
The Aerial Dataset was created using this simulator [link](https://github.com/VIS4ROB-lab/visensor_simulator).

#### 3D Models
 Most of the 3D models used to create the dataset can be download [here](https://www.polybox.ethz.ch/index.php/s/ix5FRRmi2gE8XrA). In the license files contain the authors of the 3D models. Some models were extended with a satellite image from Google Earth.

### Running the code

#### Prerequisites
* PyTorch 1.0.1
* Python 3.6
* Plus dependencies

#### Testing  Example

```bash
python3 main.py --evaluate "/media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar" --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12"
```

#### Training Example

```bash
python3 main.py --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12" --workers 8 -lr 0.00001 --batch-size 1 --dcnet-arch gudepthcompnet18 --training-mode dc1_only --criterion l2
```

```bash
python3 main.py --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12" --workers 8 --criterion l2 --training-mode dc0-cf1-ln1 --dcnet-arch ged_depthcompnet --dcnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:dc_weights --confnet-arch cbr3-c1 --confnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:conf_weights --lossnet-arch ged_depthcompnet --lossnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:lossdc_weights
```

#### Parameters

Parameter | Description
------------ | -------------
  --help            | show this help message and exit
  --output NAME       | output base name in the subfolder results
  --training-mode ARCH  | this variable indicating the training mode. Our framework has up to tree parts the dc (depth completion net), the cf (confidence estimation net) and the ln (loss net). The number 0 or 1 indicates whether the network should be updated during the back-propagation. All the networks can be pre-load using other parameters. training_mode: dc1_only ; dc1-ln0 ; dc1-ln1 ; dc0-cf1-ln0 ; dc1-cf1-ln0 ; dc0-cf1-ln1 ; dc1-cf1-ln1 (default: dc1_only)
  --dcnet-arch ARCH     | model architecture: resnet18 ; udepthcompnet18 ; gms_depthcompnet ; ged_depthcompnet ; gudepthcompnet18 (default: resnet18)
  --dcnet-pretrained PATH | path to pretraining checkpoint for the dc net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights. 
  --dcnet-modality MODALITY | modality: rgb ; rgbd ; rgbdw (default: rgbd)
  --confnet-arch ARCH   | model architecture: cbr3-c1 ; cbr3-cbr1-c1 ; cbr3-cbr1-c1res ; join ; none (default: cbr3-c1)
  --confnet-pretrained PATH | path to pretraining checkpoint for the cf net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights.
  --lossnet-arch ARCH   | model architecture: resnet18 ; udepthcompnet18 (uresnet18) ; gms_depthcompnet (nconv-ms) ; ged_depthcompnet (nconv-ed) ; gudepthcompnet18 (nconv-uresnet18) (default: ged_depthcompnet)
  --lossnet-pretrained PATH | path to pretraining checkpoint for the ln net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights.
  --data-type DATA      | dataset: visim ; kitti (default: visim)
  --data-path PATH      | path to data folder - this folder has to have inside a **val** folder and a **train** folder if it is not in evaluation mode.
  --data-modality MODALITY | this field define the input modality in the format colour-depth-weight. kfd and fd mean random sampling in the ground-truth. kgt means keypoints from slam with depth from ground-truth. kor means keypoints from SLAM with depth from the landmark. The weight can be binary (bin) or from the uncertanty from slam (kw). The parameter can be one of the following: rgb-fd-bin ; rgb-kfd-bin ; rgb-kgt-bin ; rgb-kor-bin ; rgb-kor-kw (default: rgb-fd-bin)
  --workers N     | number of data loading workers (default: 10)
  --epochs N            | number of total epochs to run (default: 15)
  --max-gt-depth D      | cut-off depth of ground truth, negative values means infinity (default: inf [m])
  --min-depth D         | cut-off depth of sparsifier (default: 0 [m])
  --max-depth D         | cut-off depth of sparsifier, negative values means infinity (default: inf [m])
  --divider D           | Normalization factor - zero means per frame (default: 0 [m])
  --num-samples N | number of sparse depth samples (default: 500)
  --sparsifier SPARSIFIER | sparsifier: uar ; sim_stereo (default: uar)
  --criterion LOSS | loss function: l1 ; l2 ; il1 (inverted L1) ; absrel (default: l1)
  --optimizer OPTIMIZER | Optimizer: sgd ; adam (default: adam)
  --batch-size BATCH_SIZE | mini-batch size (default: 8)
  --learning-rate LR | initial learning rate (default 0.001)
  --learning-rate-step LRS | number of epochs between reduce the learning rate by 10 (default: 5)
  --learning-rate-multiplicator LRM | multiplicator (default 0.1)
  --momentum M          | momentum (default: 0)
  --weight-decay W | weight decay (default: 0)
  --val-images N        | number of images in the validation image (default: 10)
  --print-freq N  | print frequency (default: 10)
  --resume PATH         | path to latest checkpoint (default: empty)
  --evaluate PATH | evaluates the model on validation set, all the training parameters will be ignored, but the input parameters still matters (default: empty)
  --precision-recall | enables the calculation of precision recall table, might be necessary to ajust the bin and top values in the ConfidencePixelwiseThrAverageMeter class. The result table shows for each confidence threshold the error and the density (default:false)
  --confidence-threshold VALUE | confidence threshold , the best way to select this number is create the precision-recall table. (default: 0)

-----------------------------------------------------------------------

#### Contact
In case of any issue, fell free to contact me via email lteixeira at mavt.ethz.ch.

