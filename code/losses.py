import torch
import torch.nn.functional as F
from torchvision import transforms


def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def energy_loss(logits, targets):
    ood_ind = 254
    void_ind = 255
    num_class = 19
    T = 1.
    m_in = -12
    m_out = -6

    energy = -(T * torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


class Gambler(torch.nn.Module):
    def __init__(self, reward, device, pretrain=-1, ood_reg=.1):
        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).cuda(device)
        self.pretrain = pretrain
        self.ood_reg = ood_reg

    def forward(self, pred, targets, wrong_sample=False):

        pred_prob = torch.softmax(pred, dim=1)

        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])
        true_pred, reservation = pred_prob[:, :-1, :, :], pred_prob[:, -1, :, :]

        # compute the reward via the energy score;
        reward = torch.logsumexp(pred[:, :-1, :, :], dim=1).pow(2)

        if not reward.shape[0] == 0:
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            reward = reward.unsqueeze(0)
            reward = gaussian_smoothing(reward)
            reward = reward.squeeze(0)
            in_reward = reward
        else:
            reward = self.reward
            in_reward = self.reward

        if wrong_sample:  # if there's ood pixels inside the image
            in_reservation = torch.div(reservation, in_reward)
            reservation = torch.div(reservation, reward)
            mask = targets == 254
            # mask out each of the ood output channel
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask.unsqueeze(1).
                repeat(1, 19, 1, 1)].log()
            
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                ood_loss = - self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            void_mask = targets == 255
            targets[void_mask] = 0  # make void pixel to 0
            targets[mask] = 0  # make ood pixel to 0
            gambler_loss = torch.gather(true_pred, index=targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, in_reservation)

            # exclude the ood pixel mask and void pixel mask
            gambler_loss = gambler_loss[(~mask) & (~void_mask)].log()
            assert not torch.any(torch.isnan(gambler_loss)), "nan check"
            return -gambler_loss.mean() + ood_loss.mean()
        else:
            mask = targets == 255
            targets[mask] = 0
            reservation = torch.div(reservation, reward)
            gambler_loss = torch.gather(true_pred, index=targets.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)

            assert torch.all(gambler_loss[~mask] > 0), "0 check"
            gambler_loss = gambler_loss[~mask].log()
            assert not torch.any(torch.isnan(gambler_loss)), "nan check"
            return -gambler_loss.mean()
