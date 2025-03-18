python -m pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install spconv-cu113 setuptools==57.5.0 wheel
python setup.py develop
pip install -e . 
pip install --upgrade onnx-simplifier==0.4.35 onnx==1.15.0 onnx_graphsurgeon==0.5.6


ln -sf /home/user/Workspace/dataset/Deeproute_open_dataset /home/user/Workspace/OpenPCDet/data/dr
python pcdet/datasets/dr/dr_dataset.py create_dr_infos  ./tools/cfgs/dataset_configs/dr_dataset.yaml 


cd tools/
python train.py --cfg_file cfgs/dr_models/pointpillar.yaml
python test.py --cfg_file ./cfgs/dr_models/pointpillar.yaml --batch_size 4 --ckpt {ckpt_file}
python generate_onnx2.py --cfg_file ./cfgs/dr_models/pointpillar.yaml --ckpt {ckpt_file}