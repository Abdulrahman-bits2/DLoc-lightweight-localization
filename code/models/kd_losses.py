"""
kd_losses.py
Feature-based knowledge-distillation losses for Study C. Self-contained and unit-tested
on synthetic tensors (no teacher/data needed).

Provides:
  - attention_transfer_loss(student_feats, teacher_feats)  -- Zagoruyko & Komodakis 2017
  - FitNetRegressor + fitnet_loss(student_feat, teacher_feat, regressor)  -- Romero 2015

Both handle mismatched channel counts and spatial resolutions between the DLoc (ResNet)
teacher and the MobileNetV2 student -- exactly the cross-architecture case Study C needs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_attention(feat, eps=1e-6):
    """Channel-collapsed spatial attention map, L2-normalized per sample.

    feat: [B, C, H, W] -> [B, H*W] with unit L2 norm (channels absorbed via sum of squares).
    No dependence on C, so teacher/student channel mismatch is irrelevant.
    """
    am = feat.pow(2).sum(dim=1)          # [B, H, W]
    am = am.flatten(1)                   # [B, H*W]
    am = am / (am.norm(p=2, dim=1, keepdim=True) + eps)
    return am


def attention_transfer_loss(student_feats, teacher_feats, eps=1e-6):
    """Mean MSE between normalized attention maps over one or more layer pairs.

    student_feats / teacher_feats: list of [B, C, H, W] tensors (paired by position).
    Teacher maps are resolution-aligned to the student map before comparison, so a
    single-stage call passes 1-element lists; multi-stage passes several.
    """
    if torch.is_tensor(student_feats):
        student_feats, teacher_feats = [student_feats], [teacher_feats]
    assert len(student_feats) == len(teacher_feats) and len(student_feats) > 0
    total = 0.0
    for fs, ft in zip(student_feats, teacher_feats):
        # align teacher spatial size to student, in 2-D, BEFORE collapsing channels
        ft2 = F.interpolate(ft.pow(2).sum(dim=1, keepdim=True), size=fs.shape[2:],
                            mode="bilinear", align_corners=False)  # [B,1,Hs,Ws]
        a_t = ft2.flatten(1)
        a_t = a_t / (a_t.norm(p=2, dim=1, keepdim=True) + eps)
        a_s = spatial_attention(fs, eps=eps)
        total = total + F.mse_loss(a_s, a_t)
    return total / len(student_feats)


class FitNetRegressor(nn.Module):
    """1x1 conv mapping student hint channels -> teacher channels (training-only add-on)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


def fitnet_loss(student_feat, teacher_feat, regressor):
    """MSE between the projected+resized student hint feature and the teacher feature."""
    s = regressor(student_feat)                                  # [B, Ct, Hs, Ws]
    s = F.interpolate(s, size=teacher_feat.shape[2:], mode="bilinear", align_corners=False)
    return F.mse_loss(s, teacher_feat)


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    # mismatched channels AND spatial sizes (teacher deeper/different res), like ResNet vs MobileNet
    s_deep = torch.randn(B, 320, 6, 12)
    t_deep = torch.randn(B, 256, 11, 23)
    s_mid = torch.randn(B, 96, 11, 23)
    t_mid = torch.randn(B, 128, 21, 46)

    l_at1 = attention_transfer_loss(s_deep, t_deep)
    l_atm = attention_transfer_loss([s_deep, s_mid], [t_deep, t_mid])
    print(f"AT single-stage loss: {l_at1.item():.6f}")
    print(f"AT multi-stage  loss: {l_atm.item():.6f}")

    reg = FitNetRegressor(320, 256)
    l_fn = fitnet_loss(s_deep, t_deep, reg)
    print(f"FitNets loss: {l_fn.item():.6f}")

    # gradients must flow to student features (and to the regressor for FitNets)
    s_deep.requires_grad_(True)
    loss = attention_transfer_loss(s_deep, t_deep) + fitnet_loss(s_deep, t_deep, reg)
    loss.backward()
    assert s_deep.grad is not None and torch.isfinite(loss)
    assert all(p.grad is not None for p in reg.parameters())
    print("gradients flow OK")
    print("OK")
