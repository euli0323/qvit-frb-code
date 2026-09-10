import argparse

import torch

from res_eval_common import (
    C_VALUES,
    evaluate_full,
    load_test_dataset,
    parse_path,
    save_json,
    summarize_c,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VIT_mix checkpoints on the full test set.")
    parser.add_argument("--data-dir", default="data/yxdata/test")
    parser.add_argument(
        "--res-train-dirs",
        nargs="+",
        default=["res_c_linear_qkv", "res_c_mlp_qkv"],
        help="One or more VIT_mix checkpoint roots to scan.",
    )
    parser.add_argument("--out-dir", default="res_test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = parse_path(args.data_dir)
    result_roots = [parse_path(item) for item in args.res_train_dirs]
    out_dir = parse_path(args.out_dir)

    print(f"[INFO] device: {device}")
    print(f"[INFO] full test data: {data_dir}")
    print(f"[INFO] result roots: {[str(path) for path in result_roots]}")
    print(f"[INFO] output dir: {out_dir}")

    test_dataset = load_test_dataset(data_dir)
    print(f"[INFO] test samples: {len(test_dataset)}")

    for c_value in C_VALUES:
        results = summarize_c(
            c_value=c_value,
            result_roots=result_roots,
            dataset=test_dataset,
            device=device,
            evaluator=lambda run_dir: evaluate_full(run_dir, test_dataset, device),
        )
        save_json(out_dir / f"res_test_c{c_value}.json", results)


if __name__ == "__main__":
    main()
