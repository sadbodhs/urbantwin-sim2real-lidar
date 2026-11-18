"""Run OpenPCDet's tools/train.py with SimDataset registered and outputs under /workspace/output.

    scripts/docker_run.sh python -m utl.train --cfg_file configs/pcdet/pp_lumpi.yaml --batch_size 8 --workers 8
"""
import runpy
import sys
from pathlib import Path

from utl.pcdet_ext import register

TOOLS = Path("/opt/OpenPCDet/tools")

def skip_non_finite_steps():
    """A batch whose total loss is not finite contributes a zero loss (zero gradients) instead of NaN weights."""
    import torch
    import pcdet.models as M
    orig = M.model_fn_decorator

    def decorator():
        f = orig()

        def model_func(model, batch_dict):
            loss, tb_dict, disp_dict = f(model, batch_dict)
            if not NAMES:   # parameter names for the F1 log (iou_reg_guards); no effect on training
                NAMES.update({id(q): n for n, q in model.named_parameters()})
            if not torch.isfinite(loss):
                print(f"non-finite loss {loss.item()}: step skipped", flush=True)
                loss = sum(p.sum() * 0 for p in model.parameters() if p.requires_grad)
            return loss, tb_dict, disp_dict
        return model_func
    M.model_fn_decorator = decorator


NAMES, STATS = {}, {"calls": 0, "skips": 0}


def iou_reg_guards(argv):
    """DSVT NaN fix (Nov 12).
    Numerical safety only, for configs with IOU_REG_LOSS: true (dsvt_lumpi, vmamba_lumpi); config, LR, seed unchanged. Any other config: no change.
      F1: OpenPCDet's train_utils.clip_grad_norm_ is wrapped: on a non-finite total norm the step is skipped (every grad set to None, logged with the
          step and up to 5 parameter names with non-finite grads). The optimizer wrapper's decoupled weight decay still applies on such a step.
      F2: training-only box decoding (centernet_utils.decode_bbox_from_pred_dicts, used only by CenterHead.get_loss) clamps the size logit to <= 6.0
          before exp(); inference decoding (decode_bbox_from_heatmap) is untouched.
      Diagnostics every 50 loss calls: max size logit over the map, count of gathered (object) size logits above 6, cumulative F1 skips."""
    import re
    if "--cfg_file" not in argv or not re.search(r"^\s*IOU_REG_LOSS:\s*[Tt]rue", Path(argv[argv.index("--cfg_file") + 1]).read_text(), re.M):
        return False
    import torch
    sys.path.insert(0, str(TOOLS))
    import train_utils.train_utils as TU
    from pcdet.models.model_utils import centernet_utils as CU
    from pcdet.models.dense_heads import center_head as CH
    orig_clip, orig_dec, orig_loss = TU.clip_grad_norm_, CU.decode_bbox_from_pred_dicts, CH.CenterHead.get_loss

    def clip(params, max_norm, *a, **k):
        params = [p for p in params if p.grad is not None]
        fin = torch.stack([torch.isfinite(p.grad).all() for p in params]) if params else None   # before clipping (clipping spreads a NaN)
        total = orig_clip(params, max_norm, *a, **k)
        if not torch.isfinite(torch.as_tensor(total)):
            bad = [NAMES.get(id(p), "?") for p, f in zip(params, fin.tolist()) if not f][:5]
            STATS["skips"] += 1
            print(f"non-finite grad: step skipped (loss call {STATS['calls']}, skips so far {STATS['skips']}) {bad}", flush=True)
            for p in params:
                p.grad = None
        return total

    def decode(pred_dict, *a, **k):
        pd = dict(pred_dict)
        pd["dim"] = pred_dict["dim"].clamp(max=6.0)
        return orig_dec(pd, *a, **k)

    def get_loss(self, *a, **k):
        STATS["calls"] += 1
        if STATS["calls"] % 50 == 1:
            with torch.no_grad():
                td = self.forward_ret_dict["target_dicts"]
                mx, above = -float("inf"), 0
                for idx, pd in enumerate(self.forward_ret_dict["pred_dicts"]):
                    d = pd["dim"]
                    mx = max(mx, float(d.max()))
                    B, C = d.shape[:2]
                    ind, m = td["inds"][idx].long(), td["masks"][idx].bool()
                    g = d.reshape(B, C, -1).gather(2, ind[:, None, :].expand(-1, C, -1))
                    above += int(((g > 6.0) & m[:, None, :]).sum())
            print(f"iou-reg guard: loss call {STATS['calls']}, max size logit over map {mx:.3f}, gathered object logits > 6: {above}, "
                  f"F1 skips so far {STATS['skips']}", flush=True)
        return orig_loss(self, *a, **k)

    TU.clip_grad_norm_ = clip
    CU.decode_bbox_from_pred_dicts = decode
    CH.CenterHead.get_loss = get_loss
    print("IOU_REG_LOSS guards on: F1 (skip non-finite-grad steps), F2 (training size logit clamp <= 6)", flush=True)
    return True


def apply_config_seed(argv):
    """A config with a top-level SEED replaces the fixed seed 666 of tools/train.py (network and data loader). Without SEED nothing changes."""
    import re
    if "--cfg_file" not in argv:
        return None
    m = re.search(r"^SEED:\s*(\d+)", Path(argv[argv.index("--cfg_file") + 1]).read_text(), re.M)
    if not m:
        return None
    seed = int(m.group(1))
    import pcdet.datasets as D
    from pcdet.utils import common_utils as C
    set_seed, build = C.set_random_seed, D.build_dataloader
    C.set_random_seed = lambda s: set_seed(s - 666 + seed)

    def build_dataloader(*args, **kwargs):
        if kwargs.get("seed") is not None:
            kwargs["seed"] = seed
        return build(*args, **kwargs)
    D.build_dataloader = build_dataloader
    print("seed from config:", seed, flush=True)
    return seed


if __name__ == "__main__":
    register()
    skip_non_finite_steps()
    apply_config_seed(sys.argv)
    iou_reg_guards(sys.argv)
    import pcdet.config
    pcdet.config.cfg.ROOT_DIR = Path("/workspace")
    sys.path.insert(0, str(TOOLS))
    sys.argv = [str(TOOLS / "train.py")] + sys.argv[1:]
    runpy.run_path(str(TOOLS / "train.py"), run_name="__main__")
