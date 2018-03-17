# KeyPatchGan

- This is a pytorch implementation of the paper, Unsupervised Holistic Image Generation from Key Local Patches. (https://arxiv.org/abs/1703.10730)


# Examples 




## Requirements
- Python2 or 3 
- Cuda device (NVIDIA GTX1080Ti was used to test)
- Pytorch
- Visdom (optional)
- Tensorflow & Tensorboard (optional)

## Preparing dataset
Download dataset via visiting [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [CompCar](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html).

For celebA dataset,
You can download using ```download.py```
```> python download.py celebA```

For compcar dataset,
Download the entire compcar dataset and some pre-processing is required.
You should crop the car patches using the ground truth bounding boxes, resize them ```128*128``` resolution, and save them in a single directory.


## Key-patches
We already extracted key patches from celebA and compcar dataset and save the bounding box coordinates to ```celebA_allbbs.mat``` and ```compcar_allbbs.mat```.
You can extract key patches and use your own key patches.


## Training celebA dataset
Run
```
python main.py --db_name=celebA --dataset_root=YOUR_DATA_ROOT --is_crop=True --image_size=108 --output_size=64 --model_structure=unet
```
The resolution of output image can be enlarged by ```--output_size=128``` or ```--output_size=256``` options.


## Training compcar dataset
Run
```
python main.py --db_name=compcar --dataset_root=YOUR_DATA_ROOT --is_crop=False --image_size=128 --output_size=128 --conv_dim=64  --batch_size=32 --model_structure=unet
```

## Misc.
Modify the options ```output_size```, ```conv_dim```, or ```batch_size``` to prevent out-of-memory error.
