import os
import re
import csv
import argparse
from glob import glob

def aggregate_eval_results(root_path, scenario_type):
    output_csv = os.path.join(root_path, f"_{scenario_type}_eval_result.csv")
    
    headers = ['category', 'Shift_AUROC']
    rows = []
    
    search_pattern = os.path.join(root_path, f"_{scenario_type}-exp_*", "eval_result.txt")
    target_files = glob(search_pattern)
    
    if not target_files:
        print(f"Warning: No eval_result.txt found for scenario '{scenario_type}' in {root_path}")
        return

    print(f"Aggregating {len(target_files)} categories for {scenario_type}...")

    for file_path in target_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        data = {'category': dir_name} 
        
        with open(file_path, 'r') as f:
            content = f.read()
            for field in headers[1:]:
                match = re.search(rf"{field}\s+([\d\.]+)", content)
                if match:
                    data[field] = match.group(1)
                else:
                    data[field] = "N/A"
        
        rows.append(data)

    rows.sort(key=lambda x: x['category'])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation results by scenario.")
    parser.add_argument("--root", type=str, required=True, help="Path to the scenario directory")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario type (A2N or N2A)")
    
    args = parser.parse_args()
    
    if args.scenario:
        aggregate_eval_results(args.root, args.scenario)
    else:
        print("Error: Scenario type is required.")