"""
studyA_configs.py
Single source of truth for the Study A run ledger + the canonical config tag.

Imported by train_studyA.py (tag), studyA_param_table.py and aggregate_studyA.py
(ledger). run_studyA.sh mirrors this ledger explicitly and MUST be kept in sync.

Ledger = 13 runs:
  - no-skip capacity curve : decoder_width {0.5, 1.0, 2.0}      (seed 42)  -> 3
  - skip   capacity curve  : decoder_width {0.5, 1.0, 2.0}      (seed 42)  -> 3
  - error-bar pair (3 seeds): {no-skip, skip} @ decoder_width 1.0          -> +4
  - multi-scale rung       : skip @ dw 1.0, bottleneck=aspp     (seed 42)  -> 1
  - bottleneck sweep       : skip @ dw 1.0, width_mult {0.5,1.4}(seed 42)  -> 2
"""

SEEDS_PAIR = [42, 123, 777]   # the 3-seed error-bar pair
SEED_MAIN = 42                # single-seed curve / exploratory points


def normalize(cfg):
    """Fill defaults for a config dict."""
    return dict(
        use_skips=bool(cfg["use_skips"]),
        decoder_width=float(cfg.get("decoder_width", 1.0)),
        width_mult=float(cfg.get("width_mult", 1.0)),
        bottleneck=cfg.get("bottleneck", "none"),
        seeds=list(cfg.get("seeds", [SEED_MAIN])),
    )


def config_tag(use_skips, decoder_width, width_mult=1.0, bottleneck="none"):
    """Canonical filename-safe tag for one architecture config (seed-independent)."""
    sk = "skip" if use_skips else "noskip"
    tag = f"{sk}_dw{float(decoder_width):g}_wm{float(width_mult):g}"
    if bottleneck != "none":
        tag += f"_{bottleneck}"
    return tag


# The ledger (architecture configs; `seeds` lists which seeds to run for each).
LEDGER = [normalize(c) for c in [
    # no-skip capacity curve
    dict(use_skips=False, decoder_width=0.5, seeds=[SEED_MAIN]),
    dict(use_skips=False, decoder_width=1.0, seeds=SEEDS_PAIR),   # error-bar pair
    dict(use_skips=False, decoder_width=2.0, seeds=[SEED_MAIN]),
    # skip capacity curve
    dict(use_skips=True, decoder_width=0.5, seeds=[SEED_MAIN]),
    dict(use_skips=True, decoder_width=1.0, seeds=SEEDS_PAIR),    # error-bar pair
    dict(use_skips=True, decoder_width=2.0, seeds=[SEED_MAIN]),
    # multi-scale rung
    dict(use_skips=True, decoder_width=1.0, bottleneck="aspp", seeds=[SEED_MAIN]),
    # bottleneck-width sweep (from-scratch, caveated)
    dict(use_skips=True, decoder_width=1.0, width_mult=0.5, seeds=[SEED_MAIN]),
    dict(use_skips=True, decoder_width=1.0, width_mult=1.4, seeds=[SEED_MAIN]),
]]


def total_runs():
    return sum(len(c["seeds"]) for c in LEDGER)


if __name__ == "__main__":
    print(f"Study A ledger: {len(LEDGER)} configs, {total_runs()} runs")
    for c in LEDGER:
        tag = config_tag(c["use_skips"], c["decoder_width"], c["width_mult"], c["bottleneck"])
        print(f"  {tag:<28} seeds={c['seeds']}")
