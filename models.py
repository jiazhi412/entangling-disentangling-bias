import torch
import torch.nn as nn
from EnD import pattern_norm


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=None, kernel_size=7, feature_pos='post'):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=True):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits

        elif self.feature_pos == 'pre':
            feats = pre_gap_feats
        elif self.feature_pos == 'post':
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats

def simple_convnet():
    model = SimpleConvNet(num_classes=10)
    model.avgpool = nn.Sequential(
        model.avgpool,
        pattern_norm()
    )
    return model

def simple_MLP(in_dim, hidden_dims, out_dim):
    model = MLP(in_dim, hidden_dims, out_dim)
    model.for_hook = nn.Sequential(
        model.for_hook,
        pattern_norm()
    )
    return model

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.for_hook = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.for_hook(x)
        return self.out(x)