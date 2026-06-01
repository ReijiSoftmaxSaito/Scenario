import os
import glob
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="結果ファイルをまとめてCSVに出力するスクリプト")
    parser.add_argument("--scenario", type=str, required=True, help="シナリオ名 (例: A2N, N2A)")
    parser.add_argument("--result_dir", type=str, default="/mnt/saito/Scenario/result",
                        help="結果ディレクトリのパス")
    parser.add_argument("--scenario_type", type=str, default="A2N",
                        choices=["A2N", "N2A", "N2A_Normal"],
                        help="検索するフォルダ名のパターン")
    parser.add_argument("--output", type=str, default="_result.csv", help="出力ファイル名")
    parser.add_argument("--mean", action="store_true", help="平均値をCSVに書き込むかどうか")
    args = parser.parse_args()

    base_dir = f"/mnt/saito/Scenario/result/_Scenario_{args.scenario}"

    if args.scenario_type.startswith("N2A"):
        result_path = os.path.join(base_dir, "N2A")
    else:
        result_path = os.path.join(base_dir, "A2N")

    if args.scenario_type.endswith("_Normal"):
        pattern = os.path.join(result_path, "_Normal-*", "_result.txt")
    else:
        pattern = os.path.join(result_path, f"_{args.scenario_type}-*", "_result.txt")

    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"該当する結果ファイルが見つかりませんでした: {os.path.join(result_path, args.scenario_type)}")
        return

    results = []
    categories = []

    for path in paths:
        category = os.path.basename(os.path.dirname(path)).replace("exp_", "")
        categories.append(category)

        data = np.loadtxt(path, dtype=str, delimiter="\t", usecols=1)
        percentages = np.array([float(d.strip('%')) for d in data])
        results.append(percentages)

    results = np.array(results)

    output_file = os.path.join(result_path, args.output)

    with open(output_file, 'w') as f:
        headers = ["Image_level_AUROC", "Pixel_level_AUROC", "PRO", "OnlyL1Score", "OtherL1Score"]
        f.write(",".join(["Category"] + headers) + "\n")

        for category, result in zip(categories, results):
            f.write(f"{category}," + ",".join([f"{x:.2f}" for x in result]) + "\n")

        if args.mean:
            mean_values = np.mean(results, axis=0)
            f.write("Mean," + ",".join([f"{x:.2f}" for x in mean_values]) + "\n")

    print(f"出力完了: {output_file}")

if __name__ == "__main__":
    main()
