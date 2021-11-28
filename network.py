# encoding: utf-8

import torch.nn as nn
from base_model.wide_network import DeepWV3Plus



class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, wide=False):
        super(Network, self).__init__()
        if wide:
            self.branch1 = DeepWV3Plus(num_classes)

    def forward(self, data, output_anomaly=False):

        return self.branch1(data, output_anomaly=output_anomaly)


