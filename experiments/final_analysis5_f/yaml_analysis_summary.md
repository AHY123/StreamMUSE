# YAML-Based StreamMUSE Analysis Report

Generated from configuration: `analysis_config.yaml`

## Experiment Configuration

### Local Direct (A4000-A4000)
- **Source**: experiments/local_direct_final
- **Description**: Direct connection between A4000 and A4000
- **Formula Type**: quadratic
- **Formula**: RT = 0.03×GL² + 17.84×GL + 12.0
- **Color**: #2E7D32
- **Samples**: 7200

### Local Server (PC-Mac)
- **Source**: experiments/local_server_final
- **Description**: Local server connection between PC client and Mac server
- **Formula Type**: quadratic
- **Formula**: RT = 0.88×GL² + 10.60×GL + 93.6
- **Color**: #1976D2
- **Samples**: 7200

### Remote Server (Hyperstack-PC)
- **Source**: experiments/remote_server_final5
- **Description**: Remote server connection between PC client and Hyperstack server
- **Formula Type**: quadratic
- **Formula**: RT = -0.19×GL² + 21.27×GL + 129.7
- **Color**: #D32F2F
- **Samples**: 7200

## Analysis Settings

- **Anomaly Filtering**: 5% per tail
- **Output Directory**: experiments/final_analysis5_f
- **Individual Analyses**: False

## Visualizations Generated

- `experiment_fitting_analysis.png`: Box plots with formula-based fitted lines and ±5% confidence bands

