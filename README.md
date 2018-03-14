# KeyPatchGan

- This is a pytorch implementation of the paper, Unsupervised Holistic Image Generation from Key Local Patches. (https://arxiv.org/abs/1703.10730)
- This is not official codes.
- Requirements: Python3, Pytorch, Cuda devices.

- To train CelebA dataset:
python main.py --db_name=celebA --dataset_root=YOUR_DATA_ROOT --is_crop=True --image_size=108 --output_size=64

- To train CompCar dataset:
python main.py --db_name=compcar_128 --dataset_root=YOUR_DATA_ROOT --is_crop=False --image_size=128 --output_size=128 --df_dim=64 --batch_size=32

You can make (128 by 128), or (256 by 256) images using --output_size option
