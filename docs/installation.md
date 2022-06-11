# Installation

The project is based on the pytorch 1.8.1 with python 3.8.

## 1. Clone the Git  repo

``` shell
$ git https://github.com/tianyu0207/PEBAL.git
$ cd pebal
```

## 2. Install dependencies

1) create conda env
    ```shell
    $ conda env create -f pebal.yml
    ```
2) install the torch 1.8.1
    ```shell
    $ conda activate pebal
    # IF cuda version < 11.0
    $ pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    # IF cuda version >= 11.0 (e.g., 30x or above)
    $ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

## 3. Prepare dataset

### cityscapes

1) please download the Cityscapes dataset (gt_Fine).
2) (optional) you might need to preprocess Cityscapes dataset
   in [here](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/preparation), as we follow the
   common setting with **19** classes.
3) specify the COCO dataset path in **code/config/config.py** file, which is **C.city_root_path**.

### coco (for outlier exposures)

1) please follow [Meta-OoD](https://github.com/robin-chan/meta-ood/tree/master/preparation) to prepare the COCO images
   serving as OoD proxy for OoD
   training. [This script](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py)
   generates binary segmentation masks for COCO images not containing any instances that could also be assigned to one
   of the Cityscapes (train-)classes. Execute via:
   ```shell 
   $ python preparation/prepare_coco_segmentation.py
   ```
2) specify the coco dataset path in **code/config/config.py** file, which is **C.coco_root_path**.

### fishyscapes

1) for the time being, you can download from the official website in [here](https://fishyscapes.com/dataset), you
   alternatively can download the preprocessed fishyscapes dataset
   in [here](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar).
2) specify the coco dataset path in **code/config/config.py** file, which is **C.fishy_root_path**.