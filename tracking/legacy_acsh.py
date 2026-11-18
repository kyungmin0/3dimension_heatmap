"""Load checkpoints trained with the team's original ACSH-YOLOv8 script.

That script defined ACmix / C2f_ACmix in its own ``__main__`` module, so the
pickled checkpoint (e.g. ``model/best.pt``) looks them up as ``__main__.ACmix``
and ``__main__.C2f_ACmix``. These classes are copied unchanged from that script
(same attribute names and forward) and injected into ``__main__`` before loading.
"""

import sys

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f


class ACmix(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False) if c1 != c2 else nn.Identity()
        self.qkv_conv = nn.Conv2d(c2, c2 * 3, kernel_size=1, stride=1, bias=False)
        self.conv_branch = nn.Conv2d(c2, c2, kernel_size=k, stride=s, padding=k // 2, groups=c2, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(c2 // 4, 8)
        self.fc1 = nn.Conv2d(c2, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, c2, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.cv1(x)
        q, k, v = torch.chunk(self.qkv_conv(x), 3, dim=1)
        f_conv = self.conv_branch(v)
        attn = torch.sigmoid(self.fc2(self.act(self.fc1(self.pool(q * k)))))
        return self.act(self.alpha * f_conv + self.beta * v * attn)


class C2f_ACmix(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.acmix = ACmix(c2, c2)

    def forward(self, x):
        return self.acmix(self.c2f(x))


def register():
    """Make the original classes resolvable for torch.load / YOLO(...)."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    main = sys.modules['__main__']
    for cls in (ACmix, C2f_ACmix):
        setattr(main, cls.__name__, cls)
    setattr(tasks, 'C2f_ACmix', C2f_ACmix)
    setattr(modules, 'C2f_ACmix', C2f_ACmix)
