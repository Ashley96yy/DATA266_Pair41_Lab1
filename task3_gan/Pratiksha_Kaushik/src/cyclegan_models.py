"""Generator / discriminator definitions, same as section 3 of Pratiksha_cyclegan_monet_v3.ipynb.

Kept in a .py file so evaluate/metrics scripts can load checkpoints without opening the notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):   # down
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):   # up
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()

        def block(cin, cout, stride, norm=True):
            l = [nn.Conv2d(cin, cout, 4, stride, 1)]
            if norm:
                l.append(nn.InstanceNorm2d(cout))
            l.append(nn.LeakyReLU(0.2, True))
            return l

        self.net = nn.Sequential(
            *block(3, ndf, 2, norm=False),
            *block(ndf, ndf * 2, 2),
            *block(ndf * 2, ndf * 4, 2),
            *block(ndf * 4, ndf * 8, 1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleD(nn.Module):
    def __init__(self, ndf=64, n_scales=1):
        super().__init__()
        self.ds = nn.ModuleList([Discriminator(ndf) for _ in range(n_scales)])

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.ds):
            if i > 0:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
            outs.append(d(x))
        return outs


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_generators(ckpt_path, device="cpu", use_ema=True):
    """Returns (G_P2M, G_M2P) from best_ema*.pt or last*.pt."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    key_p2m, key_m2p = ("EMA_P2M", "EMA_M2P") if use_ema else ("G_P2M", "G_M2P")
    g_p2m, g_m2p = Generator(), Generator()
    g_p2m.load_state_dict(ck[key_p2m])
    g_m2p.load_state_dict(ck[key_m2p])
    return g_p2m.to(device).eval(), g_m2p.to(device).eval()
