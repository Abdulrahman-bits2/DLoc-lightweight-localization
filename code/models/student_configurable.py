"""
student_configurable.py
Single parameterized student for the Study A spatial-preservation ablation.

One model, four knobs, so that ANY two configs differ ONLY by the variable being
ablated (a clean controlled comparison):

  - use_skips     : bool   U-Net skip connections on/off  -> the SPATIAL axis
  - decoder_width : float  decoder channel multiplier      -> clean CAPACITY axis
                           (encoder stays ImageNet-pretrained @ width_mult=1.0,
                            so this axis carries NO pretraining confound)
  - width_mult    : float  MobileNetV2 bottleneck width     -> SECONDARY capacity axis
                           (torchvision has pretrained weights ONLY at 1.0, so any
                            other value trains from scratch -> caveated in the paper)
  - bottleneck    : str    'none' | 'aspp'  multi-scale context module at the
                           bottleneck -> the "multi-scale features" rung

Encoder: MobileNetV2 features split into inc / enc1[1:4] / enc2[4:7] / enc3[7:14] /
enc4[14:-1] -- the SAME index ranges as student_unet.py, so Study A results are directly
comparable to the shipped U-Net run. Stage channel counts are detected dynamically at
init (robust to width_mult).

Decoder: 3 x (Upsample x2 -> [concat skip] -> Conv-BN-ReLU) -> 1x1 -> Sigmoid, then
interpolate to (161, 361). use_skips=False simply drops the three concats -- the ONLY
structural difference between the skip and no-skip families.

Reference points:
  ConfigurableStudent(use_skips=True,  decoder_width=1.0) ~= shipped U-Net   (~2.41M, 0.640 m)
  ConfigurableStudent(use_skips=False, decoder_width=1.0) ~= plain MobileNet (~2.3M,  0.707 m)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPPLite(nn.Module):
    """Lightweight multi-scale context module (ASPP-style), channel-preserving.

    Parallel 1x1 + dilated 3x3 (small rates, since the bottleneck map is ~6x12) +
    global-average-pool branch, concatenated and projected back to `in_ch` so the
    decoder is unchanged. This is the "+ multi-scale features" rung of the ablation.
    """

    def __init__(self, in_ch, rates=(1, 2, 4)):
        super().__init__()
        # Keep the module light so its parameter cost does not confound the
        # "multi-scale helps independent of params" reading (branch = in_ch // 4).
        branch_ch = max(8, in_ch // 4)
        self.b1x1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * n_branches, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [self.b1x1(x)] + [d(x) for d in self.dilated]
        g = self.gap(x)
        g = F.interpolate(g, size=size, mode='bilinear', align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class ConfigurableStudent(nn.Module):
    OUTPUT_SIZE = (161, 361)

    def __init__(self, input_nc=4, use_skips=True, decoder_width=1.0,
                 width_mult=1.0, bottleneck='none'):
        super().__init__()
        assert bottleneck in ('none', 'aspp'), f'bad bottleneck: {bottleneck}'
        self.use_skips = bool(use_skips)
        self.cfg = dict(input_nc=input_nc, use_skips=self.use_skips,
                        decoder_width=decoder_width, width_mult=width_mult,
                        bottleneck=bottleneck)

        # Pretrained ImageNet weights exist ONLY at width_mult == 1.0.
        pretrained = abs(float(width_mult) - 1.0) < 1e-6
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mnet = mobilenet_v2(weights=weights, width_mult=float(width_mult))

        # First conv: 3-ch RGB -> input_nc-ch AoA/ToF (fresh, pretrained can't map 4ch).
        old = mnet.features[0][0]
        first = nn.Conv2d(input_nc, old.out_channels, kernel_size=old.kernel_size,
                          stride=old.stride, padding=old.padding, bias=False)
        self.inc = nn.Sequential(first, mnet.features[0][1], mnet.features[0][2])

        # Encoder stages (same index ranges as student_unet.py).
        self.enc1 = mnet.features[1:4]     # skip 1
        self.enc2 = mnet.features[4:7]     # skip 2
        self.enc3 = mnet.features[7:14]    # skip 3
        self.enc4 = mnet.features[14:-1]   # bottleneck (drop final 320->1280 conv)

        # Detect channel counts dynamically (robust to width_mult).
        with torch.no_grad():
            d = torch.zeros(1, input_nc, *self.OUTPUT_SIZE)
            x0 = self.inc(d)
            s1 = self.enc1(x0)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            b = self.enc4(s3)
        c_s1, c_s2, c_s3, c_b = s1.shape[1], s2.shape[1], s3.shape[1], b.shape[1]

        # Multi-scale context module (channel-preserving) or a no-op.
        self.bottleneck = ASPPLite(c_b) if bottleneck == 'aspp' else nn.Identity()

        # Decoder channel schedule, scaled by decoder_width.
        base = [128, 64, 32]
        dch = [max(8, int(round(c * float(decoder_width)))) for c in base]
        skip_ch = [c_s3, c_s2, c_s1] if self.use_skips else [0, 0, 0]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = _conv_block(c_b + skip_ch[0], dch[0])
        self.dec2 = _conv_block(dch[0] + skip_ch[1], dch[1])
        self.dec3 = _conv_block(dch[1] + skip_ch[2], dch[2])
        self.final = nn.Sequential(nn.Conv2d(dch[2], 1, kernel_size=1), nn.Sigmoid())

        # Remember skip tensors' channels for the forward concat.
        self._skip_channels = (c_s3, c_s2, c_s1)

    def _maybe_cat(self, d, skip):
        if not self.use_skips:
            return d
        skip = F.interpolate(skip, size=d.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([d, skip], dim=1)

    def forward(self, x):
        x0 = self.inc(x)
        s1 = self.enc1(x0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.enc4(s3)
        b = self.bottleneck(b)

        d = self.dec1(self._maybe_cat(self.up(b), s3))
        d = self.dec2(self._maybe_cat(self.up(d), s2))
        d = self.dec3(self._maybe_cat(self.up(d), s1))

        out = self.final(d)
        if out.shape[2:] != self.OUTPUT_SIZE:
            out = F.interpolate(out, size=self.OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return out


def build_student(use_skips=True, decoder_width=1.0, width_mult=1.0,
                  bottleneck='none', input_nc=4):
    """Convenience constructor mirroring the CLI knobs of train_studyA.py."""
    return ConfigurableStudent(
        input_nc=input_nc, use_skips=use_skips, decoder_width=decoder_width,
        width_mult=width_mult, bottleneck=bottleneck,
    )


if __name__ == "__main__":
    # Shape + param sanity check for a few configs.
    configs = [
        dict(use_skips=False, decoder_width=1.0),
        dict(use_skips=True, decoder_width=1.0),
        dict(use_skips=True, decoder_width=0.5),
        dict(use_skips=True, decoder_width=2.0),
        dict(use_skips=True, decoder_width=1.0, bottleneck='aspp'),
        dict(use_skips=True, decoder_width=1.0, width_mult=0.5),
        dict(use_skips=True, decoder_width=1.0, width_mult=1.4),
    ]
    x = torch.randn(2, 4, 161, 361)
    for cfg in configs:
        m = build_student(**cfg)
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 1, 161, 361), f"Wrong shape {y.shape} for {cfg}"
        print(f"{cfg}  ->  {n/1e6:.3f}M params, out {tuple(y.shape)}")
    print("OK")
