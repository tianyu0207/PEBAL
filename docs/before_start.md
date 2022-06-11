# Getting Started

we visualize our training details via wandb (https://wandb.ai/site).

## visualization

1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or add the key to the "code/config/config.py" with
   ```shell
   C.wandb_key = ""
   ```

## training

our code is trained using one nvidia 3090 GPU, but our code also supports distributed data parallel mode in pytorch. We
set batch_size=8 for all the experiments, with learning rate 1e-5 and 900 * 900 resolution.

### checkpoints

we follow [Meta-OoD](https://github.com/robin-chan/meta-ood) and use the deeplabv3+ checkpoint
in [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet).
**NOTE: Please download the checkpoint to "pretrained_ckpts" before running the code.**

for training, simply execute

```shell 
$ python code/main.py 
```

### inference

please download our checkpoint
from [here](https://drive.google.com/file/d/12CebI1TlgF724-xvI3vihjbIPPn5Icpm/view?usp=sharing) and specify the
checkpoint path in config file.

```shell
python code/test.py
```