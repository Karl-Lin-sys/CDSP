"""Original teaching baseline made from official torch modules; not a paper reproduction."""
import torch
from torch import nn
from torch.nn import functional as F

class TinyRestorer(nn.Module):
    def __init__(self, channels=3, width=16, scale=1):
        """O(W) parameter allocation; scale=1 denoising, 2/4 joint denoising and SR."""
        super().__init__()
        if channels not in (1, 3) or width < 1 or scale not in (1, 2, 4):
            raise ValueError("Unsupported model configuration")
        self.channels, self.width, self.scale = channels, width, scale
        self.net = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, channels*scale*scale, 3, padding=1),
                                 nn.PixelShuffle(scale))

    def forward(self, x):
        """O(B*H*W*9*(C*D+D^2+D*C*s^2)); memory O(B*H*W*(D+C*s^2))."""
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return base + self.net(x)
