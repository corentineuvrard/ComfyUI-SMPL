"""Fit the (height_cm, weight_kg) -> betas regressors and print the fit report.

The node pack fits the regressor automatically on first use; this script does
the same thing explicitly, for every installed model (or the ones selected),
and prints the fit errors. The procedure is fully synthetic and deterministic
(see smpl_avatar/regressor.py and docs/TECHNICAL.md, 'Shape regressor').

Usage::

    python scripts/train_regressor.py
    python scripts/train_regressor.py --model-types SMPL --genders female --samples 50000 --seed 1
    python scripts/train_regressor.py --report report.json --output my_regressor.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from smpl_avatar import body_model, paths, regressor  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-types", nargs="+", default=list(body_model.MODEL_TYPES),
                        choices=body_model.MODEL_TYPES)
    parser.add_argument("--genders", nargs="+", default=list(body_model.GENDERS), choices=body_model.GENDERS)
    parser.add_argument("--samples", type=int, default=regressor.DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=regressor.DEFAULT_SEED)
    parser.add_argument("--beta-std", type=float, default=regressor.DEFAULT_BETA_STD)
    parser.add_argument("--output", type=Path, default=paths.REGRESSOR_PATH,
                        help=f"regressor file to update (default: {paths.REGRESSOR_PATH})")
    parser.add_argument("--report", type=Path, help="also write the full fit report as JSON")
    args = parser.parse_args(argv)

    stored = {}
    if args.output.is_file():
        try:
            stored = regressor.load_regressors(args.output)
        except Exception as exc:  # another format version, a corrupt file, ...
            print(f"warning: ignoring the existing {args.output} ({type(exc).__name__}: {exc}); "
                  "it will be replaced")
    reports = {}
    for model_type in args.model_types:
        for gender in args.genders:
            try:
                body = body_model.load(model_type, gender)
            except FileNotFoundError:
                print(f"skip {model_type} {gender}: model file not installed")
                continue
            except body_model.BodyModelFormatError as exc:
                print(f"skip {model_type} {gender}: {exc}")
                continue
            start = time.perf_counter()
            reg = regressor.train(
                body.v_template, body.shapedirs, body.faces, gender=gender,
                samples=args.samples, seed=args.seed, beta_std=args.beta_std,
                extra_meta={"model_type": model_type, "model_file": body.path.name},
            )
            key = regressor.regressor_key(model_type, gender)
            stored[key] = reg
            reports[key] = reg.meta
            print(f"\n{model_type} {gender}  ({time.perf_counter() - start:.1f} s, "
                  f"training heights {reg.meta['training_height_range_cm'][0]:.0f}-"
                  f"{reg.meta['training_height_range_cm'][1]:.0f} cm, weights "
                  f"{reg.meta['training_weight_range_kg'][0]:.0f}-{reg.meta['training_weight_range_kg'][1]:.0f} kg)")
            print(f"  {'evaluation set':<18}{'height MAE':>12}{'height max':>12}{'weight MAE':>12}{'weight max':>12}{'weight MAPE':>13}")
            for set_name in ("held_out", "grid_bmi_18.5_30", "grid"):
                e = reg.meta["report"][set_name]
                if e is None:
                    continue
                print(f"  {set_name:<18}{e['height_mae_cm']:>9.2f} cm{e['height_max_cm']:>9.2f} cm"
                      f"{e['weight_mae_kg']:>9.2f} kg{e['weight_max_kg']:>9.2f} kg{e['weight_mape_pct']:>11.2f} %")
            r2 = reg.meta["report"]["beta_r2_held_out"]
            print("  held-out R^2 per beta: " + " ".join(f"{x:.2f}" for x in r2))

    if not reports:
        print("No usable body model installed; nothing to do.")
        return 1
    regressor.save_regressors(args.output, stored)
    print(f"\nSaved {len(reports)} regressor(s) to {args.output}")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2, sort_keys=True))
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
