import argparse

import torch

from res_eval_common import (
    Q_VALUES,
    SNR_THRESHOLD,
    evaluate_snr,
    extract_snr,
    load_test_dataset,
    parse_path,
    save_json,
    summarize_q,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QVIT_mix2 checkpoints on low/high SNR test subsets."
    )
    parser.add_argument("--data-dir", default="data/yxdata/test")
    parser.add_argument(
        "--res-train-dirs",
        nargs="+",
        default=["res_q_distill"],
        help="One or more QVIT_mix2 checkpoint roots to scan.",
    )
    parser.add_argument("--out-dir", default="res_test_snr")
    parser.add_argument("--snr-threshold", type=float, default=SNR_THRESHOLD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = parse_path(args.data_dir)
    result_roots = [parse_path(item) for item in args.res_train_dirs]
    out_dir = parse_path(args.out_dir)

    print(f"[INFO] device: {device}")
    print(f"[INFO] test data: {data_dir}")
    print(f"[INFO] result roots: {[str(path) for path in result_roots]}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] SNR threshold: {args.snr_threshold}")

    base_dataset = load_test_dataset(data_dir)
    low_count = sum(1 for path, _ in base_dataset.samples if extract_snr(path) < args.snr_threshold)
    high_count = len(base_dataset) - low_count
    print(f"[INFO] test samples: {len(base_dataset)}")
    print(f"[INFO] low samples : {low_count}")
    print(f"[INFO] high samples: {high_count}")

    for q_value in Q_VALUES:
        results = summarize_q(
            q_value=q_value,
            result_roots=result_roots,
            dataset=base_dataset,
            device=device,
            evaluator=lambda run_dir: evaluate_snr(
                run_dir,
                base_dataset,
                device,
                args.snr_threshold,
            ),
        )
        save_json(out_dir / f"res_test_snr_q{q_value}.json", results)


if __name__ == "__main__":
    main()
