# StreamMUSE Bulk Analysis System

A comprehensive analysis tool for comparing generation length (GL) experiments across multiple datasets. This system can handle both manually-run benchmarks (individual CSV files) and bulk benchmark results (from `bulk_benchmark.py`).

## Features

- **Multi-experiment comparison**: Analyze and compare results from multiple experiment directories
- **Dual format support**: Works with both old manual data and new bulk benchmark data
- **Generation length focus**: Specialized analysis for generation length (GL) parameter studies
- **Comprehensive visualizations**: Comparative plots across all experiments
- **Individual analysis**: Optional detailed analysis for each experiment
- **Unified reporting**: Combined summary with experiment comparisons

## Quick Start

1. **Basic usage - compare multiple experiment directories**:
   ```bash
   python app/analysis/bulk_analysis.py experiments/exp1_results experiments/exp2_results experiments/exp3_results
   ```

2. **With custom output directory**:
   ```bash
   python app/analysis/bulk_analysis.py experiments/*/results --output_dir my_comparison_analysis
   ```

3. **Fast comparison (skip individual analyses)**:
   ```bash
   python app/analysis/bulk_analysis.py experiments/*/results --skip_individual
   ```

4. **With anomaly filtering (remove outliers)**:
   ```bash
   python app/analysis/bulk_analysis.py experiments/exp1_results experiments/exp2_results --anomaly_filter 5.0
   ```

## Input Data Formats

### Format 1: Bulk Benchmark Results
From `bulk_benchmark.py` output directories containing:
- `combined_results.csv` - All experiment results
- `config.yaml` - Experiment configuration
- `experiment_summary.json` - Experiment metadata

Example directory structure:
```
experiments/my_bulk_experiment/
├── combined_results.csv
├── config.yaml
├── experiment_summary.json
├── my_experiment_GL1_PL96.csv
└── my_experiment_GL5_PL96.csv
```

### Format 2: Manual Benchmark Results
Directories with individual CSV files where generation length can be extracted from filenames:

Supported filename patterns:
- `gen_length_5.csv`
- `gen_5.csv` 
- `generation_5.csv`
- `GL5.csv`
- `5_frames.csv`
- `frames_5.csv`

Example directory structure:
```
experiments/manual_experiment/
├── gen_length_1.csv
├── gen_length_3.csv
├── gen_length_5.csv
└── gen_length_7.csv
```

## Command Line Options

```bash
python app/analysis/bulk_analysis.py [directories...] [options]

Required:
  directories              List of experiment directories to analyze

Options:
  --output_dir DIR        Output directory for analysis results (default: bulk_analysis_results)
  --skip_individual       Skip individual experiment analyses (faster, comparative plots only)
  --no_plots             Skip all plot generation (data processing only)
  --anomaly_filter FLOAT  Filter out anomalies: percentage of data to remove from each tail (0-50)
  -h, --help             Show help message
```

## Output Structure

The bulk analysis creates a comprehensive output directory:

```
bulk_analysis_results/
├── bulk_analysis_summary.md           # Main summary report
├── plots/                             # Comparative visualizations
│   ├── experiment_comparison.png      # Side-by-side experiment comparison
│   ├── generation_length_trends.png   # Scaling trends across experiments
│   ├── performance_comparison.png     # Bar chart performance comparison
│   ├── variability_comparison.png     # Variability analysis
│   └── constraint_analysis.png        # Real-time constraint analysis
├── processed_data/                    # Combined datasets
│   ├── all_experiments_combined.csv   # All raw data combined
│   ├── all_experiments_summary.csv    # Summary statistics
│   ├── constraint_analysis.csv        # Detailed constraint analysis results
│   └── parameter_constraint_analysis.csv  # I vs GL validity matrices
└── individual_experiment_name/        # Individual analyses (if not skipped)
    ├── plots/                         # Detailed plots for this experiment
    ├── summary_table.csv              # Summary statistics
    └── summary_report.txt             # Text summary
```

## Key Visualizations

### 1. Experiment Comparison (`experiment_comparison.png`)
- Side-by-side comparison of mean and 95th percentile round trip time
- Shows how different experiments perform across generation lengths
- Identifies which experimental conditions perform best

### 2. Generation Length Trends (`generation_length_trends.png`)  
- Four-panel plot showing scaling trends across experiments
- Includes round trip time, inference duration, server processing, and network latency
- Shows trend lines for each experiment to identify scaling patterns

### 3. Performance Comparison (`performance_comparison.png`)
- Bar chart comparison for key generation lengths (1, 3, 5, 7, 9 frames)
- Side-by-side bars for different experiments
- Useful for identifying optimal generation length for each experimental condition

### 4. Constraint Analysis (`constraint_analysis.png`)
- Analyzes which experiments meet real-time musical constraints
- Shows musical time deadlines (125ms per tick at 120 BPM)
- Compares both mean and 95th percentile performance against constraints

### 6. Detailed Constraint Analysis (`detailed_constraint_analysis.png`)
- Four-panel analysis of constraint satisfaction rates
- Shows overall satisfaction by generation length across experiments
- Compares 95th percentile constraint satisfaction
- Breaks down constraint types (overlap vs buffer violations)
- Analyzes satisfaction rates across different RTT percentiles

### 7. Constraint Heatmaps (`constraint_heatmaps.png`)
- Per-experiment heatmaps showing valid/invalid parameter combinations
- Tests inference interval vs generation length parameter space
- Uses 95th percentile RTT for robustness analysis
- Shows which parameter combinations meet real-time constraints

### 8. Parameter Constraint Analysis (`parameter_constraint_analysis.png`)
- Comprehensive I vs GL validity matrices across all experiments and percentiles
- Grid layout showing constraint satisfaction for each experiment and percentile level
- Color-coded constraint status: ✓=Valid, C1=Overlap Violation, C2=Buffer Violation, ✗=Both Violated
- Individual per-experiment plots (`parameter_constraints_{experiment_name}.png`)

### 9. Individual Parameter Constraint Plots
- Detailed 2x3 grid showing all 6 percentiles (50th, 60th, 70th, 80th, 90th, 99.5th)
- Clear visualization of parameter space validity for each experimental condition
- Annotated cells showing constraint violation types

### 5. Variability Comparison (`variability_comparison.png`)
- Compares latency variability across experiments
- Shows both absolute (standard deviation) and relative (coefficient of variation) variability
- Helps identify which experimental conditions are most consistent

## Anomaly Filtering

Both analysis tools support anomaly filtering to remove outliers that might skew results:

### How It Works
- **Per-generation-length filtering**: For each generation length (GL), removes data points from both tails of the round trip time distribution
- **Symmetric filtering**: Removes equal percentages from both high and low extremes within each GL
- **Independent filtering**: Each generation length is filtered separately to preserve relative performance characteristics

### Usage Examples
```bash
# Remove top and bottom 2.5% per generation length (5% total per GL)
python app/analysis/analyze_generation_length_results.py data.csv --anomaly_filter 2.5

# Remove top and bottom 5% per generation length (10% total per GL)
python app/analysis/bulk_analysis.py experiments/*/results --anomaly_filter 5.0

# Remove top and bottom 10% per generation length (20% total per GL)
python app/analysis/bulk_analysis.py experiments/*/results --anomaly_filter 10.0
```

### When to Use
- **High variability**: When you see extreme outliers in your data
- **Network issues**: When some requests experienced unusual network delays
- **System load**: When the server had varying load during benchmarking
- **Comparative studies**: When comparing experiments where some may have more noise

### Recommendations
- **Conservative approach**: Start with 2.5-5% for robust analysis while preserving most data
- **Aggressive filtering**: Use 10%+ only when outliers are clearly non-representative
- **Document filtering**: Always note the filtering percentage in your analysis reports

### Output Information
When anomaly filtering is applied, the tools provide detailed feedback per generation length:
```
🔍 Anomaly filtering (5.0% each tail per generation length):
   GL 1: 200 → 180 samples (removed 20, 10.0%)
      RTT range: 45.2ms - 78.3ms
   GL 3: 200 → 180 samples (removed 20, 10.0%) 
      RTT range: 52.1ms - 95.7ms
   GL 5: 200 → 180 samples (removed 20, 10.0%)
      RTT range: 61.3ms - 123.7ms
   Total: 600 → 540 samples (removed 60, 10.0%)
```

## Analysis Focus: Generation Length Studies

This tool is specifically designed for analyzing how **generation length** (number of frames generated per request) affects system performance. Key aspects:

### Musical Time Constraints
- **Tick-based analysis**: Focuses on odd generation lengths (accompaniment frames)
- **Real-time deadlines**: 125ms per tick at 120 BPM
- **Buffer analysis**: Shows how much musical time is available for processing

### Performance Metrics
- **Round trip time**: Total request latency including network
- **Inference duration**: Time spent in model inference
- **Server processing**: Total server-side processing time
- **Network latency**: Estimated network overhead

### Scaling Analysis
- **Linear scaling**: How latency increases with generation length
- **Efficiency**: Notes generated per second vs generation length
- **Variability**: How consistent performance is across generation lengths

## Example Use Cases

### 1. Parameter Optimization Study
Compare experiments with different prompt lengths:
```bash
python app/analysis/bulk_analysis.py \
  experiments/prompt_24_results \
  experiments/prompt_48_results \
  experiments/prompt_96_results \
  --output_dir prompt_length_comparison
```

### 2. Model Comparison Study  
Compare different model sizes or configurations:
```bash
python app/analysis/bulk_analysis.py \
  experiments/model_small_results \
  experiments/model_medium_results \
  experiments/model_large_results \
  --output_dir model_size_comparison
```

### 3. Hardware Comparison Study
Compare performance on different hardware:
```bash
python app/analysis/bulk_analysis.py \
  experiments/gpu_v100_results \
  experiments/gpu_a100_results \
  experiments/cpu_only_results \
  --output_dir hardware_comparison
```

### 4. Mixed Data Analysis
Combine old manual data with new bulk benchmark results:
```bash
python app/analysis/bulk_analysis.py \
  legacy_data/manual_experiment_1 \
  legacy_data/manual_experiment_2 \
  new_data/bulk_experiment_results \
  --output_dir legacy_vs_new_comparison
```

## Integration with Existing Tools

### Relationship to Other Analysis Scripts
- **`analyze_generation_length_results.py`**: Used internally for individual experiment analysis
- **`parameter_constraint_analysis.py`**: Complementary constraint analysis tool
- **`bulk_benchmark.py`**: Generates compatible input data

### Data Pipeline Integration
```bash
# 1. Run bulk experiments
python app/benchmarking/bulk_benchmark.py experiments/my_config.yaml

# 2. Run bulk analysis on results
python app/analysis/bulk_analysis.py experiments/my_experiment_results --output_dir analysis

# 3. Optional: Run constraint analysis on combined data
python app/analysis/parameter_constraint_analysis.py analysis/processed_data/all_experiments_combined.csv
```

## Best Practices

### 1. Experiment Organization
- Use descriptive directory names that indicate experimental conditions
- Keep experiment configurations for reproducibility
- Document experimental parameters in directory names or metadata

### 2. Data Quality
- Ensure sufficient sample sizes (>= 50 requests per generation length)
- Use consistent benchmark parameters across experiments being compared
- Validate that generation lengths overlap between experiments

### 3. Analysis Workflow
- Start with `--skip_individual` for quick comparative analysis
- Run full analysis with individual experiments for detailed investigation
- Use constraint analysis to validate real-time performance requirements

### 4. Interpretation
- Focus on both mean and 95th percentile performance for robustness
- Consider variability alongside absolute performance
- Validate findings against musical timing constraints

## Troubleshooting

### Common Issues

1. **No generation length data found**:
   - Check filename patterns match supported formats
   - Ensure CSV files have appropriate columns (`generation_length` or `generation_length_frames`)
   - Verify directory contains benchmark result files

2. **Mixed data format errors**:
   - Ensure all experiments have consistent column naming
   - Check for missing required columns in manual data
   - Validate that bulk benchmark data has `combined_results.csv`

3. **Memory issues with large datasets**:
   - Use `--skip_individual` to reduce memory usage
   - Process experiments in smaller batches
   - Ensure sufficient system memory for large combined datasets

4. **Plot generation errors**:
   - Use `--no_plots` to skip visualization if only data processing is needed
   - Check for missing dependencies (matplotlib, seaborn)
   - Ensure sufficient display capability for plot generation

### Debug Mode
For troubleshooting data loading issues, the script provides detailed feedback:
- Lists each experiment directory being processed
- Shows which files are found and loaded successfully
- Indicates data format detection (bulk vs manual)
- Reports total requests and generation lengths found

## Future Enhancements

Potential improvements for future versions:
- Support for additional parameter dimensions (prompt length, tempo, etc.)
- Statistical significance testing between experiments
- Automated optimal parameter recommendation
- Export to standard analysis formats (Excel, JSON, etc.)
- Integration with experiment tracking systems