import argparse
import torch.optim
from valid import *
from utils.logger import *
from engine.engine import Engine
from config.config import config
from model.network import Network
from collections import OrderedDict
from engine.evaluator import SlidingEval
from dataset.data_loader import Fishyscapes, Cityscapes
from utils.img_utils import Compose, Normalize, ToTensor

# warnings.filterwarnings('ignore', '.*imshow.*', )


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
    logger = logging.getLogger("pebal")
    logger.propagate = False

    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])

    cityscapes = Cityscapes(root=config.city_root_path, split="val", transform=transform)
    evaluator = SlidingEval(config, device=0 if engine.local_rank < 0 else engine.local_rank)
    fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path, transform=transform)
    fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=transform)

    # we only support 1 gpu for testing
    model = Network(config.num_classes, wide=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=engine.devices)
    model.to(device)
    engine.load_pebal_ckpt(config.pebal_weight_path, model=model)

    model.eval()
    """
    # 1). we currently only support single gpu valid for the cityscapes sliding validation, and it 
    # might take long time, feel free to uncomment it. (we'll have to use the sliding eval. to achieve 
      the performance reported in the GitHub. )
    # 2). we follow Meta-OoD to use single scale validation for OoD dataset, for fair comparison.
    """
    valid_epoch(model=model, engine=engine, test_set=cityscapes, my_wandb=None,
                evaluator=evaluator, logger=logger)

    valid_anomaly(model=model, epoch=0, test_set=fishyscapes_ls, data_name='Fishyscapes_ls',
                  my_wandb=None, logger=logger)

    valid_anomaly(model=model, epoch=0, test_set=fishyscapes_static,
                  data_name='Fishyscapes_static', my_wandb=None, logger=logger)


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

    args.world_size = args.nodes * args.gpus
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))
