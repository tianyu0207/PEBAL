import argparse
import torch.distributed as dist
import torch.optim
from config.config import config
from dataset.data_loader import Fishyscapes, Cityscapes
from dataset.data_loader import get_mix_loader
from engine.engine import Engine
from engine.evaluator import SlidingEval
from engine.lr_policy import WarmUpPolyLR
from engine.trainer import Trainer
from losses import *
from model.network import Network
from utils.img_utils import *
from utils.wandb_upload import *
from valid import *

from utils.logger import *

warnings.filterwarnings('ignore', '.*imshow.*', )


def declare_settings(config_file, logger, engine):
    logger.critical("distributed data parallel training: {}".format(str("on" if engine.distributed is True
                                                                        else "off")))
    
    logger.critical("gpus: {}, with batch_size[local]: {}".format(engine.world_size, config.batch_size))

    logger.critical("network architecture: {}, with ResNet {} backbone".format("deeplabv3+",
                                                                               config_file['pretrained_weight_path']
                                                                               .split('/')[-1].split('_')[0]))
    logger.critical("learning rate: other {}, and head is same [world]".format(config_file['lr']))

    logger.info("image: {}x{} based on 1024x2048".format(config_file['image_height'],
                                                         config_file['image_width']))

    logger.info("current batch: {} [world]".format(int(config_file['batch_size']) * engine.world_size))


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    logger = logging.getLogger("pebal")
    logger.propagate = False
    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    if engine.local_rank <= 0:
        declare_settings(config_file=config, logger=logger, engine=engine)
        visual_tool = Tensorboard(config=config)
    else:
        visual_tool = None

    seed = config.seed

    if engine.distributed:
        seed = seed + engine.local_rank

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = Network(config.num_classes, wide=True)
    gambler_loss = Gambler(reward=[4.5], pretrain=-1, device=engine.local_rank if engine.local_rank >= 0 else 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    testing_transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])
    fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path, transform=testing_transform)
    fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=testing_transform)
    cityscapes = Cityscapes(root=config.city_root_path, split="val", transform=testing_transform)

    # config lr policy
    base_lr = config.lr
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    trainer = Trainer(engine=engine, loss1=gambler_loss, loss2=energy_loss, lr_scheduler=lr_policy,
                      ckpt_dir=config.saved_dir, tensorboard=visual_tool)

    evaluator = SlidingEval(config, device=0 if engine.local_rank < 0 else engine.local_rank)

    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)
        model.cuda(engine.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                          find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=engine.devices)
        model.to(device)

    # starting with the pre-trained weight from https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet
    if engine.continue_state_object:
        engine.register_state(dataloader=None, model=model, optimizer=optimizer)
        engine.restore_checkpoint(extra_channel=True)
        # engine.load_pebal_ckpt(config.pebal_weight_path, model=model)

    logger.info('training begin...')

    for curr_epoch in range(engine.state.epoch, config.nepochs):

        train_loader, train_sampler, void_ind = get_mix_loader(engine=engine, augment=True,
                                                               cs_root=config.city_root_path,
                                                               coco_root=config.coco_root_path)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)

        trainer.train(model=model, epoch=curr_epoch, train_sampler=train_sampler, train_loader=train_loader,
                      optimizer=optimizer)

        if curr_epoch % config.eval_epoch == 0:
            if engine.local_rank <= 0:
                """
                # 1). we currently only support single gpu valid for the cityscapes sliding validation, and it 
                # might take long time, feel free to uncomment it. (we'll have to use the sliding eval. to achieve 
                  the performance reported in the GitHub. )
                # 2). we follow Meta-OoD to use single scale validation in OoD datasets, for fair comparison.
                """
                # valid_epoch(model=model, engine=engine, test_set=cityscapes, my_wandb=visual_tool,
                #             evaluator=evaluator, logger=logger)

                valid_anomaly(model=model, epoch=curr_epoch, test_set=fishyscapes_ls, data_name='Fishyscapes_ls',
                              my_wandb=visual_tool, logger=logger)

                valid_anomaly(model=model, epoch=curr_epoch, test_set=fishyscapes_static,
                              data_name='Fishyscapes_static', my_wandb=visual_tool, logger=logger)

        if engine.distributed:
            dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=1,
                        type=int,
                        help="gpus in use")
    parser.add_argument('-l', '--local_rank', default=-1,
                        type=int,
                        help="distributed or not")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int,
                        help="distributed or not")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    args.world_size = args.nodes * args.gpus

    # we enforce the flag of ddp if gpus >= 2;
    args.ddp = True if args.world_size > 1 else False
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))
