import torch.optim
import argparse
import torch.backends.cudnn as cudnn
from dataloader import get_test_loader
from network import Network
from dataloader import Fishyscapes
from imageaugmentations import Compose, Normalize, ToTensor
from evaluation import *
from collections import OrderedDict
warnings.filterwarnings('ignore', '.*imshow.*', )

def get_anomaly_detector(num_classes, criterion=None):
  """
  Get Network Architecture based on arguments provided
  """
  ckpt_name = 'best_ad_ckpt.pth'
  model = Network(num_classes, criterion=criterion, norm_layer=torch.nn.BatchNorm2d, wide=True)

  tmp = torch.load(ckpt_name)
  print('################ retore ckpt from {} #############################'.format(ckpt_name))
  state_dict = tmp
  if 'model' in state_dict.keys():
    state_dict = state_dict['model']
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k
    new_state_dict[name] = v
  state_dict = new_state_dict
  model.load_state_dict(state_dict, strict=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  return model


def main(gpu, ngpus_per_node, config, args):

  args.local_rank = gpu
  transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])
  Fishyscapes_ls = Fishyscapes(split='LostAndFound', root="/home/yu/yu_ssd/fishyscapes/", transform=transform)
  Fishyscapes_static = Fishyscapes(split='Static', root="/home/yu/yu_ssd/fishyscapes/", transform=transform)

  model = get_anomaly_detector(20)

  print('Validating begin...')

  valid_anomaly(model=model, test_set=Fishyscapes_static, data_name='Fishyscapes_static')

  valid_anomaly(model=model, test_set=Fishyscapes_ls, data_name='Fishyscapes_ls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=1,
                        type=int,
                        help="gpus in use")
    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")
    parser.add_argument('-l', '--local_rank', default=-1,
                        type=int,
                        help="distributed or not")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int,
                        help="distributed or not")

    args = parser.parse_args()
    cudnn.benchmark = True

    args.world_size = args.nodes * args.gpus
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))


