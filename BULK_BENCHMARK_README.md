# StreamMUSE Bulk Benchmark System

A comprehensive benchmarking system that allows running multiple benchmark experiments with different parameter combinations using YAML configuration files.

## Features

- **Grid Search**: Automatically generate all combinations of specified parameters
- **Multiple Experiments**: Define different experiment sets within one configuration
- **Result Aggregation**: Combine results from all experiments into unified datasets
- **Automatic Analysis**: Generate statistics and summaries
- **Error Handling**: Retry failed experiments automatically
- **Flexible Configuration**: YAML-based configuration with validation

## Quick Start

1. **Start your StreamMUSE server**:
   ```bash
   CHECKPOINT_PATH=path/to/model.ckpt uvicorn app.server:app --host 0.0.0.0 --port 8000
   ```

2. **Run a quick test**:
   ```bash
   python app/bulk_benchmark.py experiments/quick_test_config.yaml
   ```

3. **Preview experiments without running them**:
   ```bash
   python app/bulk_benchmark.py experiments/bulk_benchmark_config.yaml --dry-run
   ```

## Configuration File Structure

### Basic Structure
```yaml
experiment:
  name: "my_experiment"
  description: "Description of what this experiment tests"
  output_dir: "experiments/my_experiment_results"

server:
  url: "http://localhost:8000/generate_accompaniment"

benchmark:
  num_requests: 100
  tempo: 120.0

parameters:
  generation_length_frames: [1, 3, 5, 7, 9]
  prompt_length_ticks: [48, 96, 144]
```

### Grid Search Parameters

The `parameters` section defines the grid search space. Each parameter can be:
- **Single value**: `generation_length_frames: 5`
- **List of values**: `generation_length_frames: [1, 3, 5, 7, 9]`

Available parameters:
- `generation_length_frames`: Number of frames to generate per request
- `prompt_length_ticks`: Context length in ticks for the model
- `tempo`: BPM for musical timing analysis
- `assumed_network_latency_ms`: Additional network latency (ms)
- `inference_interval_ticks`: Specific tick interval to analyze

### Multiple Experiment Sets

You can define multiple experiment groups:
```yaml
parameters:
  generation_length_frames: [1, 3, 5]
  prompt_length_ticks: [96]

additional_experiments:
  - name: "prompt_sensitivity"
    description: "Test prompt length sensitivity"
    parameters:
      generation_length_frames: [5]
      prompt_length_ticks: [24, 48, 72, 96, 120, 144]
  
  - name: "tempo_analysis"
    description: "Test different tempos"
    parameters:
      generation_length_frames: [5]
      prompt_length_ticks: [96]
      tempo: [60, 90, 120, 150, 180]
```

### Execution Options

```yaml
execution:
  parallel: false  # Run experiments in parallel (not recommended for single server)
  max_workers: 1   # Number of parallel workers
  retry_failed: true  # Retry failed experiments
  max_retries: 2
  delay_between_experiments: 5.0  # Seconds between experiments
  clear_server_history: true  # Clear server history between experiments
```

### Analysis Options

```yaml
analysis:
  generate_plots: true  # Generate visualization plots
  generate_summary: true  # Generate experiment summary
  export_combined_csv: true  # Combine all results into one CSV
  calculate_percentiles: [50, 70, 80, 90, 95, 99]
```

### Musical Injection Options

```yaml
injection:
  enabled: true  # Enable musical prompt injection
  file_path: "path/to/prompt.mid"  # MIDI file to inject as prompt
  # injection_length_ticks automatically matches prompt_length_ticks parameter
```

When injection is enabled:
- Before each experiment, the specified MIDI file is injected into the server's history
- The injection length automatically matches the `prompt_length_ticks` parameter
- Benchmark requests are automatically adjusted to account for the injection offset
- This ensures the model has musical context that matches the tested prompt length

## Output Structure

Each experiment run creates a structured output directory:

```
experiments/my_experiment_results/
├── config.yaml                    # Copy of configuration used
├── experiment_summary.json        # Metadata and results summary
├── combined_results.csv          # All experiments in one CSV
├── statistics_summary.txt        # Basic statistics
├── my_experiment_gen_1_prompt_48.csv
├── my_experiment_gen_1_prompt_96.csv
├── my_experiment_gen_3_prompt_48.csv
└── ...                           # Individual experiment CSV files
```

## Example Configurations

### 1. Generation Length Study
Test how generation length affects performance:
```yaml
parameters:
  generation_length_frames: [1, 3, 5, 7, 9, 11, 13, 15]
  prompt_length_ticks: [96]  # Fixed context length
```

### 2. Context Length Study  
Test how context length affects performance:
```yaml
parameters:
  generation_length_frames: [5]  # Fixed generation length
  prompt_length_ticks: [24, 48, 72, 96, 120, 144, 168, 192]
```

### 3. Full Grid Search
Test all combinations of key parameters:
```yaml
parameters:
  generation_length_frames: [1, 3, 5, 7, 9]
  prompt_length_ticks: [48, 96, 144]
  tempo: [100, 120, 140]
# This creates 5 × 3 × 3 = 45 experiments
```

## Analysis and Results

### Combined Results CSV
All experiment results are combined into `combined_results.csv` with columns:
- All original benchmark columns (request_id, round_trip_time, etc.)
- Parameter columns (generation_length_frames, prompt_length_ticks, etc.)
- Experiment metadata (experiment_name, experiment_duration)

### Statistics Summary
`statistics_summary.txt` contains:
- Overall statistics across all experiments
- Round trip time percentiles
- Statistics grouped by parameter combinations

### Experiment Summary JSON
`experiment_summary.json` contains:
- Experiment configuration
- Success/failure counts
- Individual experiment metadata
- Failed experiment details

## Integration with Analysis Tools

The bulk benchmark results are compatible with existing analysis tools:

```bash
# Analyze bulk benchmark results
python app/analyze_generation_length_results.py experiments/my_experiment_results/combined_results.csv --output_dir analysis_results

# Run constraint analysis on bulk results
python app/parameter_constraint_analysis.py experiments/my_experiment_results/combined_results.csv
```

## Best Practices

1. **Start Small**: Use `quick_test_config.yaml` to verify setup before large experiments
2. **Server Stability**: Ensure server is stable before running long experiments
3. **Resource Management**: Monitor system resources during large experiments
4. **Backup Configs**: Keep configuration files in version control
5. **Result Organization**: Use descriptive experiment names and output directories

## Troubleshooting

### Common Issues

1. **Server Connection Errors**:
   - Verify server is running and accessible
   - Check server URL in configuration
   - Ensure no firewall blocking connections

2. **Failed Experiments**:
   - Check server logs for errors
   - Verify parameter combinations are valid
   - Use retry functionality for transient failures

3. **Memory Issues**:
   - Reduce batch size (num_requests)
   - Add delays between experiments
   - Monitor server memory usage

### Debug Mode
Use `--dry-run` to preview experiments without running them:
```bash
python app/bulk_benchmark.py config.yaml --dry-run
```

This shows all planned experiments and their parameters without executing them.