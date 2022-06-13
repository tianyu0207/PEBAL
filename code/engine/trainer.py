import torch
from tqdm import tqdm


class Trainer:
    """
    loss_1 -> gambler loss; loss_2 -> energy loss
    lr_scheduler -> cosine;
    """

    def __init__(self, engine, loss1, loss2, tensorboard, lr_scheduler=None, ckpt_dir=None):
        self.engine = engine
        self.loss1 = loss1
        self.loss2 = loss2
        self.lr_scheduler = lr_scheduler
        self.saved_dir = ckpt_dir
        self.tensorboard = tensorboard

    def train(self, model, epoch, train_sampler, train_loader, optimizer):
        model.train()

        self.freeze_model_parameters(model)

        if self.engine.distributed:
            train_sampler.set_epoch(epoch)

        loader_len = len(train_loader)
        tbar = tqdm(range(loader_len), ncols=137, leave=True) if self.engine.local_rank <= 0 else range(loader_len)
        train_loader = iter(train_loader)

        for batch_idx in tbar:
            minibatch = next(train_loader)
            optimizer.zero_grad()
            curr_idx = epoch * loader_len + batch_idx

            self.engine.update_iteration(epoch, curr_idx)

            imgs = minibatch['data'].cuda(non_blocking=True)
            target = minibatch['label'].cuda(non_blocking=True)
            is_ood = minibatch['is_ood']

            logits = model(imgs)
            in_logits, in_target = logits[~is_ood], target[~is_ood]
            out_logits, out_target = logits[is_ood], target[is_ood]

            e_loss, _ = self.loss2(logits=logits, targets=target)

            loss = self.loss1(pred=in_logits, targets=in_target, wrong_sample=False)

            if torch.any(is_ood):
                loss += self.loss1(pred=out_logits, targets=out_target, wrong_sample=True)

            loss += 0.1 * e_loss

            loss.backward()
            optimizer.step()

            # update learning rate
            current_lr = self.lr_scheduler.get_lr(cur_iter=curr_idx)
            for _, opt_group in enumerate(optimizer.param_groups):
                opt_group['lr'] = current_lr

            curr_info = {}
            if self.engine.local_rank <= 0:
                curr_info['gambler_loss'] = loss
                curr_info['energy_loss'] = e_loss * .1
                self.tensorboard.upload_wandb_info(current_step=curr_idx, info_dict=curr_info)

        if self.engine.local_rank <= 0:
            self.engine.save_and_link_checkpoint(snapshot_dir=self.saved_dir, name='epoch_{}.pth'.format(epoch))

        return

    @staticmethod
    def freeze_model_parameters(curr_model):
        for name, param in curr_model.named_parameters():
            if 'module.branch1.final' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
