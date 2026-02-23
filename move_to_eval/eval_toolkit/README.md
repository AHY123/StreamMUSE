# Evaluation Toolkit (评估工具箱)

This module provides a suite of generic utilities for processing, analyzing, and exporting evaluation metrics from StreamMUSE experiments. It is designed to be modular and reusable across different metric types (Results, Raw Experiments, NLL data).
本模块为处理、分析和导出 StreamMUSE 实验的评估指标提供了一套通用工具。它设计为模块化架构，可在不同的指标类型（客观结果、原始实验数据、NLL数据）之间复用。

## Structure (模块结构)

- **`path_utils.py`**: Path manipulation and string utilities. Defines the registry of metric types (`RESULT`, `EXP_RAW`, `NLL`) and provides helpers like `get_path` to reliably map experiment keys to their respective data directories or files.
  *(路径处理与字符串工具。定义了指标类型的注册表，并提供了如 `get_path` 等辅助函数，用于可靠地将实验 key 映射到各自的数据目录或文件。)*
- **`json_parser.py`**: JSON parsing helpers for different metric types. Extracts underlying metrics from complex JSON structures (lists of dicts, nested metadata) and flattens them into normalized lists of target values ready for statistical analysis.
  *(针对不同指标类型的 JSON 解析辅助工具。从复杂的 JSON 结构中提取底层指标数据，并将其展平为标准的目标值列表，以便进行统计分析。)*
- **`stats.py`**: Numeric and statistical helpers designed to be dependency-free (relying only on the standard `statistics` library). Exposes `compute_stats` which returns comprehensive descriptive statistics (mean, variance, min, max, percentiles, IQR, etc.) and is robust against missing or `NaN` values.
  *(数值与统计辅助工具，无第三方依赖（仅依赖标准 `statistics` 库）。提供 `compute_stats` 方法返回全面的描述性统计特征（均值、方差、最值、百分位数、四分位距等），且能够稳健地处理缺失值。)*
- **`csv_exporter.py`**: A CLI tool to export aggregated statistics from parsed JSON files into a structurally clean CSV format. Suitable for downstream pivot tables or analysis.
  *(一个命令行工具，用于将解析后的 JSON 统计数据导出为结构清晰的 CSV 格式，适用于后续的数据透视或分析。)*

## Usage (使用示例)

The components of this toolkit are typically imported and orchestrated together by caller scripts (e.g., `run.py`, `run_all_types.py`).
该工具箱的组件通常由调用方脚本（如 `run.py`, `run_all_types.py`）在一起引入并协同工作。

### Exporting Statistics to CSV (导出统计数据至 CSV)

To run the unified exporter which automatically processes multiple metrics into a single sheet:
*(运行统一的可执行脚本，将多种实验指标处理并汇总到单张工作表中：)*

```bash
python -m move_to_eval.eval_toolkit.csv_exporter \
  --base_dir result/results-experiments2-local \
  --out summary_report.csv \
  --types pitch_jsd,onset_jsd,consonant_ratio \
  --stats mean,stdev_samp
```

### Direct Programmatic Usage (代码中直接调用)

```python
from move_to_eval.eval_toolkit.path_utils import get_path
from move_to_eval.eval_toolkit.json_parser import parse_by_type
from move_to_eval.eval_toolkit.stats import compute_stats

key = "interval4_gen5_prompt_128_gen_576"

# 1. Resolve path (解析路径)
p = get_path(key, "hit_rate", "result/results-experiments2-local")

# 2. Extract specific values (提取具体指标数值)
items = parse_by_type(key, "hit_rate", p)

# 3. Compute structural statistics (计算结构化统计信息)
stats = compute_stats(items)
print(f"Mean Hit Rate: {stats.get('mean')}")
```
