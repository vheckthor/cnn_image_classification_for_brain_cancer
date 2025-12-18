"""
Model comparison utilities.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd
from evaluation.visualization import plot_metrics_comparison


def generate_comparison_report(
    results_path: Path,
    output_dir: Path
):
    """
    Generate comprehensive comparison report.
    
    Args:
        results_path: Path to evaluation results JSON
        output_dir: Directory to save report
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    if 'custom' in results and 'pretrained' in results:
        comparison_data = []
        
        for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
            custom_val = results['custom']['metrics'].get(metric, 0)
            pretrained_val = results['pretrained']['metrics'].get(metric, 0)
            diff = pretrained_val - custom_val
            
            comparison_data.append({
                'Metric': metric.upper(),
                'Custom Model': custom_val,
                'Pre-trained Model': pretrained_val,
                'Difference': diff,
                'Winner': 'pretrained' if diff > 0 else 'custom'
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save as CSV
        csv_path = output_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"Comparison table saved to: {csv_path}")
        
        # Save as markdown
        md_path = output_dir / "comparison_report.md"
        with open(md_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"**Overall Winner**: {results.get('overall_winner', 'N/A')}\n\n")
            f.write("## Metrics Comparison\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Custom Model Best Val DSC: {results['custom'].get('best_val_dice', 'N/A')}\n")
            f.write(f"- Pre-trained Model Best Val DSC: {results['pretrained'].get('best_val_dice', 'N/A')}\n")
            f.write(f"- Custom Model Epochs: {results['custom'].get('epoch', 'N/A')}\n")
            f.write(f"- Pre-trained Model Epochs: {results['pretrained'].get('epoch', 'N/A')}\n")
        
        print(f"Comparison report saved to: {md_path}")
        
        # Create visualization
        plot_path = output_dir / "metrics_comparison.png"
        plot_metrics_comparison(
            results['custom']['metrics'],
            results['pretrained']['metrics'],
            save_path=plot_path
        )
    
    print(f"\nComparison report generated in: {output_dir}")

