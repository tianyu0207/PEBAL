# PEBAL
This repo contains the Pytorch implementation of our paper:
> [**Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes**](https://arxiv.org/pdf/2111.12264.pdf)
>
> [Yu Tian*](https://yutianyt.com/), Yuyuan Liu*, [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), Fengbei Liu, Yuanhong Chen, [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **SOTA on 4 benchmarks.** Check out [**Papers With Code**](https://paperswithcode.com/paper/pixel-wise-energy-biased-abstention-learning). 

![image](https://user-images.githubusercontent.com/19222962/161691512-61a2dfa8-2079-465c-abaa-5b8fdf42e5f7.png)


## Inference

> [**Checkpoint for anomaly segmentation**](https://drive.google.com/file/d/12CebI1TlgF724-xvI3vihjbIPPn5Icpm/view?usp=sharing)

After downloading the pre-trained checkpoint, simply run the following command: 
```shell
python test.py
```

**Training Code will be released soon**

## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@misc{tian2021pixelwise,
      title={Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes}, 
      author={Yu Tian and Yuyuan Liu and Guansong Pang and Fengbei Liu and Yuanhong Chen and Gustavo Carneiro},
      year={2021},
      eprint={2111.12264},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
---
