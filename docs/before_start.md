# Getting Started

we visualize our training details via wandb (https://wandb.ai/site).

## visualization

1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you can find you API key in (https://wandb.ai/authorize)

3) add the key to the "code/config/config.py" with
   ```shell
   C.wandb_key = ""
   ```

## training

our code is trained using one nvidia 3090 GPU, but our code also supports distributed data parallel mode in pytorch. We
set batch_size=8 for all the experiments, with learning rate 1e-5 and 900 * 900 resolution.

### checkpoints

we follow [Meta-OoD](https://github.com/robin-chan/meta-ood) and use the deeplabv3+ checkpoint
in [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet). you'll need to put it in "ckpts/pretrained_ckpts" directory, and
**please note that downloading the checkpoint before running the code is necessary for our approach.**

for training, simply execute

```shell 
$ python code/main.py 
```

### inference

please download our checkpoint
from [here](https://drive.google.com/file/d/12CebI1TlgF724-xvI3vihjbIPPn5Icpm/view?usp=sharing) and specify the
checkpoint path ("ckpts/pebal_weight_path") in config file.

```shell
python code/test.py
```
