#!/usr/bin/python
"""
studyC_layer_probe.py
Pre-flight for Study C: print the teacher/student feature shapes that the KD losses use,
so the layer mapping is confirmed on-instance (the one documented judgment call).

Student (always loadable):  encoder bottleneck + a mid stage.
Teacher (needs DLoc weights): encoder bottleneck (enc_model.output) + output heatmap
(teacher.decoder.output) -- the two reliable signals train_studyC.py distils from.

Usage (on the instance, after data+weights are present):
  python studyC_layer_probe.py --teacher_dir ./runs/2026-05-21-06-51-03
"""

# --- path bootstrap ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import torch


def probe_student():
    from student_mobilenet import MobileNetStudent
    student = MobileNetStudent(input_nc=4).eval()
    feats = {}
    student.encoder.register_forward_hook(lambda m, i, o: feats.__setitem__("deep", o))
    student.encoder[13].register_forward_hook(lambda m, i, o: feats.__setitem__("mid", o))
    with torch.no_grad():
        student(torch.zeros(1, 4, 161, 361))
    print("STUDENT (MobileNetStudent):")
    print(f"  encoder bottleneck 'deep' : {tuple(feats['deep'].shape)}  (expect ~[1,320,6,12])")
    print(f"  encoder[13] mid           : {tuple(feats['mid'].shape)}   (expect ~[1,96,11,23])")
    return feats


def probe_teacher(teacher_dir):
    try:
        from modelADT import ModelADT
        from joint_model import Enc_2Dec_Network
        from params_storage.params_fig10b_single_gpu import (
            opt_exp, opt_encoder, opt_decoder, opt_offset_decoder)
    except Exception as e:
        print(f"TEACHER: import failed ({e}). Run this on the instance with the DLoc env.")
        return
    for opt in (opt_encoder, opt_decoder, opt_offset_decoder):
        opt.checkpoints_load_dir = teacher_dir
    enc = ModelADT(); enc.initialize(opt_encoder); enc.setup(opt_encoder)
    dec = ModelADT(); dec.initialize(opt_decoder); dec.setup(opt_decoder)
    off = ModelADT(); off.initialize(opt_offset_decoder); off.setup(opt_offset_decoder)
    teacher = Enc_2Dec_Network()
    teacher.initialize(opt_exp, enc, dec, off, gpu_ids=opt_exp.gpu_ids)
    for m in (enc, dec, off):
        m.load_networks("best", load_dir=teacher_dir)
    teacher.eval()

    # one forward on a dummy batch (wo, w, lbl all zeros)
    z = torch.zeros(1, 4, 161, 361)
    lbl = torch.zeros(1, 1, 161, 361)
    with torch.no_grad():
        teacher.set_input(z, lbl, z, shuffle_channel=False)
        teacher.test()
    print("TEACHER (DLoc):")
    print(f"  enc_model.output 'deep'   : {tuple(enc.output.shape)}   (bottleneck; FitNets/AT target)")
    print(f"  decoder.output heatmap    : {tuple(teacher.decoder.output.shape)}  (output-KD + multi-AT target)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="./runs/2026-05-21-06-51-03")
    args = ap.parse_args()
    probe_student()
    print()
    probe_teacher(args.teacher_dir)


if __name__ == "__main__":
    main()
