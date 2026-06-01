import os
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="A2N_Normal 結果集計")
parser.add_argument("--method", type=str, required=True, help="手法名 (例: FastFlow, VQ, INP_Former)")
parser.add_argument("--result_dir", type=str, default="/mnt/saito/Scenario/result")
parser.add_argument("--scenario_type", type=str, default="A2N_Normal", choices=["A2N_Normal"])
parser.add_argument("--mean", action="store_true", help="平均値を追加するかどうか")
args = parser.parse_args()

result_path = os.path.join(args.result_dir, f"_Scenario_{args.method}", "A2N")

paths = sorted(glob.glob(os.path.join(result_path, "*Normal*", "*_result.txt")))
if not paths:
    print(f"該当する結果ファイルが見つかりませんでした: {result_path}")
    exit()

columns = ["Category", "Image_level_AUROC", "Pixel_level_AUROC", "PRO", "OnlyL1Score", "OtherL1Score"]
df_scores = pd.DataFrame(columns=columns)

for path in paths:
    category_prefix = os.path.basename(os.path.dirname(path)).replace("exp_", "").replace("_Normal-", "").replace("None", "")
    with open(path, 'r') as file:
        auroc_values = {}
        abnormal_values = {}
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            metric_full_name, value = parts[0], parts[1].strip('%')
            try:
                score = float(value)
            except:
                continue

            if metric_full_name in ["Image_level_AUROC", "Pixel_level_AUROC", "PRO"]:
                auroc_values[metric_full_name] = score
            elif "OnlyL1Score" in metric_full_name or "OtherL1Score" in metric_full_name:
                abnormal_type, score_type = metric_full_name.split('-')
                if abnormal_type not in abnormal_values:
                    abnormal_values[abnormal_type] = {}
                abnormal_values[abnormal_type][score_type] = score

        for abnormal_type, scores in abnormal_values.items():
            row = {
                "Category": f"{category_prefix}-{abnormal_type}",
                "Image_level_AUROC": auroc_values.get("Image_level_AUROC", None),
                "Pixel_level_AUROC": auroc_values.get("Pixel_level_AUROC", None),
                "PRO": auroc_values.get("PRO", None),
                "OnlyL1Score": scores.get("OnlyL1Score", None),
                "OtherL1Score": scores.get("OtherL1Score", None)
            }
            df_scores = pd.concat([df_scores, pd.DataFrame([row])], ignore_index=True)

if args.mean:
    mean_row = pd.DataFrame([{
        "Category": "Mean",
        "Image_level_AUROC": round(df_scores["Image_level_AUROC"].mean(), 2),
        "Pixel_level_AUROC": round(df_scores["Pixel_level_AUROC"].mean(), 2),
        "PRO": round(df_scores["PRO"].mean(), 2),
        "OnlyL1Score": round(df_scores["OnlyL1Score"].mean(), 2),
        "OtherL1Score": round(df_scores["OtherL1Score"].mean(), 2)
    }])
    df_scores = pd.concat([df_scores, mean_row], ignore_index=True)

output_file = os.path.join(result_path, f"_result_{args.scenario_type}.csv")
df_scores.to_csv(output_file, index=False)
print("出力完了:", output_file)
