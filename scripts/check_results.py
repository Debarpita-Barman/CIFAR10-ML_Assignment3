#!/usr/bin/env python3
"""Check that results/ contains every file the report needs, and print the headline test numbers.

Usage: python scripts/check_results.py [--results results]
Exit code 1 if anything is missing.
"""
import argparse
import json
from pathlib import Path

EXPECTED = {
    "q1_outputs": ["results_summary.json", "split_indices.npz", "interpretation.txt", "environment_used.json",
                   "pip_freeze.txt", "table_a1_model_specs.csv", "table_a2_lr_selection.csv",
                   "table_a3_validation_summary.csv", "table_a4_evidence.csv", "table_final_test_q1a.csv",
                   "table_b1_memorisation_trajectory.csv", "table_b2_stable_learning_time.csv",
                   "table_final_test_q1b.csv", "fig_a0_permutation_examples.png", "fig_a1_curves_factorial.png",
                   "fig_a2_curves_capacity.png", "fig_a3_capacity_and_factorial.png", "fig_b1_memorisation.png"],
    "q2_outputs": ["results_summary.json", "conclusion.txt", "environment_used.json", "table_q2_models.csv",
                   "table_q2_transformations.csv", "table_q2_validation.csv", "table_q2_test_results.csv",
                   "table_q2_confidence.csv", "table_q2_per_class_accuracy.csv", "table_q2_run_sources.csv",
                   "fig_q2_0_conditions.png", "fig_q2_1_curves_clean.png", "fig_q2_2_first_layer_weights.png",
                   "fig_q2_3_curves_perm.png", "fig_q2_4_val_curves_all.png", "fig_q2_5_tile_curve.png",
                   "fig_q2_6_reliability.png", "fig_q2_7_confidence_hist.png", "fig_q2_8_per_class.png",
                   "fig_q2_9_confusion_cnn.png"],
    "q3_outputs": ["results_summary.json", "conclusion.txt", "environment_used.json",
                   "table_q3a_autoencoders_validation.csv", "table_q3a_test_reconstruction.csv",
                   "table_q3b_test_classification.csv", "table_q3c_test_representations.csv",
                   "fig_q3_1_ae_training.png", "fig_q3_2_reconstructions.png", "fig_q3_3_denoising_examples.png",
                   "fig_q3_4_severity.png", "fig_q3_5_recon_vs_probe.png"],
}
EXECUTED = ["Q1_MLP_CIFAR10.ipynb", "Q2_CNN_vs_MLP_CIFAR10.ipynb", "Q3_Autoencoders_CIFAR10.ipynb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = Path(ap.parse_args().results)
    missing = []
    for folder, files in EXPECTED.items():
        for f in files:
            if not (res / folder / f).exists():
                missing.append(f"{folder}/{f}")
    for f in EXECUTED:
        if not (res / "executed" / f).exists():
            missing.append(f"executed/{f}  (executed notebook = run log)")
    seeds = set()
    for folder in EXPECTED:
        p = res / folder / "results_summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            seeds.add(s.get("seed"))
            if folder == "q1_outputs":
                t = s["test_q1a"]
                print(f"Q1 test acc: MLP-M {100 * sum(t['MLP-M']) / len(t['MLP-M']):.1f}%, "
                      f"CNN {100 * sum(t['CNN']) / len(t['CNN']):.1f}%")
            if folder == "q2_outputs":
                print("Q2 CNN retention R:", {k: round(v, 3) if v is not None else None for k, v in s["retention_R"].items()})
            if folder == "q3_outputs":
                print(f"Q3 selected d={s['selected_d']}, recovered fraction F={s['recovered_fraction']}, "
                      f"Spearman={s['spearman_linear']}")
    if len(seeds) > 1:
        missing.append(f"INCONSISTENT SEEDS across questions: {seeds}")
    if missing:
        print(f"{len(missing)} missing item(s):")
        for m in missing:
            print("  -", m)
        raise SystemExit(1)
    print("All expected result files are present; seed", seeds.pop() if seeds else "?")


if __name__ == "__main__":
    main()
