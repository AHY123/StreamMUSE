from scripts.dir_stats import get_path
from scripts.json_processors import parse_by_type
from scripts.aggreate import compute_stats
import os

# path = get_path("interval1_gen3_prompt_128_gen_576", "nll", "result/results-experiments2-local")

# print(path)
# print(os.path.isdir(path))  # Check if the path is a directory
# print(os.path.isfile(path))  # Check if the path is a file

# p = get_path("interval1_gen3_prompt_128_gen_576", "consonant_ratio", "result/results-experiments2-local")
# items = parse_by_type("interval1_gen3_prompt_128_gen_576", "consonant_ratio", p)
# # items 是一个列表，包含 detail 中对应的数据
# print(len(items))
# print(items)  # 打印解析后的数据

# # Compute and print simple statistics for the extracted items.
# stats = compute_stats(items)
# missing = len(items) - stats.get("count", 0)
# print("\nStatistics:")
# print(f"  total_items: {len(items)}")
# print(f"  missing_items (None/unparsed): {missing}")
# for k in ("count", "sum", "min", "max", "mean", "p25", "p50", "p75", "iqr", "variance_pop", "variance_samp"):
#     print(f"  {k}: {stats.get(k)!r}")


# type = "backup_level"
type = "hit_rate"
key = "interval4_gen5_prompt_128_gen_576"
p = get_path(key, type, "result/results-experiments2-local")
# print(p)
# print(os.path.isdir(p))  # Check if the path is a directory
# print(os.path.isfile(p))  # Check if the path is a file
items = parse_by_type(key, type, p)
# items 是一个列表，包含 detail 中对应的数据
# print(len(items))
# print(items)  # 打印解析后的数据

# Compute and print simple statistics for the extracted items.
stats = compute_stats(items)
missing = len(items) - stats.get("count", 0)
print("\nStatistics:")
print(f"  total_items: {len(items)}")
print(f"  missing_items (None/unparsed): {missing}")
for k in ("count", "sum", "min", "max", "mean", "p25", "p50", "p75", "iqr", "variance_pop", "variance_samp"):
    print(f"  {k}: {stats.get(k)!r}")
