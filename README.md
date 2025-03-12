# SMNet

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
# step 1. clone this repo
cd SMNet

# step 2. create a conda virtual environment and activate it
conda create -n pointenet python=3.7 -y
conda activate pointenet
conda install pytorch torchvision cudatoolkit
pip install cycler einops h5py pyyaml scikit-learn scipy tqdm matplotlib
pip install pointnet2_ops_lib/.
```

### Dataset
Please download the following datasets: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), [ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip), and [ShapeNetPart](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Then, create a `data/` folder and organize the datasets as
```
PointeNet
|--classification_ModelNet40
|---- data/
|––---- modelnet40_ply_hdf5_2048/
|--classification_ScanObjectNN
|---- data/
|––---- h5_files/
|--part_segmentation
|---- data/
|––---- shapenetcore_partanno_segmentation_benchmark_v0_normal/
```

## Useage
### Shape Classification
For the ModelNet40 dataset, just run:
```bash
cd classification_ModelNet40
python main.py --model SMNet --msg <output filename>
```

For the ScanObjectNN dataset, just run:
```bash
cd classification_ScanObjectNN
python main.py --model SMNet --msg <output filename>
```

### Part Segmentation
For the ShapeNetPart dataset, just run:
```bash
python main.py --model SMNet --exp_name <output filename>
```





