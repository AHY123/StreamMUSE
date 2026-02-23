import os
import shutil

os.makedirs("move_to_eval/nll_compute", exist_ok=True)
if os.path.exists("scripts/eval_toolkit") and not os.path.exists("move_to_eval/eval_toolkit"):
    shutil.move("scripts/eval_toolkit", "move_to_eval/eval_toolkit")

for f in ["cal_nll.py", "cal_nll_aggregate.py", "plot_nll_heatmap.py", "run_nll_from_manifest.py"]:
    if os.path.exists(f):
        shutil.move(f, f"move_to_eval/nll_compute/{f}")

legacy = ["aggreate.py", "stats_utils.py", "json_processors.py", "utils.py", "dir_stats.py", "export_results_csv.py"]
for f in legacy:
    path = f"scripts/{f}"
    if os.path.exists(path):
        os.remove(path)

print("Migration completed successfully")
