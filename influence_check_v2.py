import json
import glob
import re
import os
import openpyxl

# base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5-2lvl-orig"
# filepattern = "*_system_*_results_gpt-5-2025-08-07_run*_2026-04-06.json"
# reg_formula = r"(.+?)_system_(.+?)_results_gpt-5-2025-08-07_run(\d+)_2026-04-06\.json"
# output = f"/mnt/d/Naved/analysis/prompt_influence/gpt5-2lvl-orig_gpt-5-2025-08-07_all_results_2026-04-06.xlsx"

# base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5-orig"
# filepattern = "*_system_*_results_gpt-5-2025-08-07_run*_2026-*-*.json"
# reg_formula = r"(.+?)_system_(.+?)_results_gpt-5-2025-08-07_run(\d+)_2026-(.+?)-(.+?)\.json"
# output = f"/mnt/d/Naved/analysis/prompt_influence/gpt5-orig_gpt-5-2025-08-07_all_results.xlsx"

# base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5.4-2lvl-orig"
# filepattern = "*_system_*_results_gpt-5.4-2026-03-05_run*_2026-04-*.json"
# reg_formula = r"(.+?)_system_(.+?)_results_gpt-5.4-2026-03-05_run(\d+)_2026-04-(.+?)\.json"
# output = f"/mnt/d/Naved/analysis/prompt_influence/gpt5.4-2lvl-orig_gpt-5.4-2026-03-05_all_results.xlsx"

# base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5-2lvl-orig-scr5"
# filepattern = "*_system_*_results_gpt-5-2025-08-07_run*_2026-04-15.json" 
# reg_formula = r"(.+?)_system_(.+?)_results_gpt-5-2025-08-07_run(\d+)_2026-04-15\.json"
# output = f"/mnt/d/Naved/analysis/prompt_influence/gpt5-2lvl-orig-scr5_gpt-5-2025-08-07_all_results.xlsx"

base = "/mnt/d/Naved/Outputs/prompt_influence/gpt5.4-2lvl-orig-scr5"
filepattern = "*_system_*_results_gpt-5.4-2026-03-05_run*_2026-04-15.json" 
reg_formula = r"(.+?)_system_(.+?)_results_gpt-5.4-2026-03-05_run(\d+)_2026-04-15.json"
output = f"/mnt/d/Naved/analysis/prompt_influence/gpt5.4-2lvl-orig-scr5_gpt-5.4-2026-03-05_all_results.xlsx"

#----------
rows = []
for path in glob.glob(f"{base}/" + filepattern):
    fname = os.path.basename(path)
    m = re.match(reg_formula, fname)
    if not m:
        continue
    sys_prompt = "Yes" if m.group(1) != "no" else "No"
    context_prompt = m.group(2)
    run = int(m.group(3))
    with open(path) as f:
        data = json.load(f)
    for entry in data["predictions"]:
        isic = re.search(r"ISIC_(\d+)\.jpg", entry["file"])
        if not isic:
            continue
        rows.append([isic.group(1), entry["true_label"], context_prompt, sys_prompt, run, entry["score"], entry["rationale"]])

rows.sort(key=lambda x: (x[0], x[2], x[3], x[4]))

wb = openpyxl.Workbook()
ws = wb.active
ws.append(["ISIC", "true_label", "ContextPrompt", "SysPrompt", "Run#", "Score", "Explanation"])
for row in rows:
    ws.append(row)

wb.save(output)
print(f"Extracted {len(rows)} rows from {len(glob.glob(f'{base}/' + filepattern))} files")
