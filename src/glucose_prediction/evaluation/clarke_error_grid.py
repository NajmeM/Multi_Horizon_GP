"""
Clarke Error Grid analysis for glucose prediction evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def clarke_error_grid(
    ref_values: np.ndarray, 
    pred_values: np.ndarray, 
    title_string: str = "Clarke Error Grid"
) -> Tuple[plt.Figure, List[int]]:
    """
    Generates a Clarke Error Grid for diabetes glucose predictions.
    
    Parameters:
    -----------
    ref_values : array-like
        The reference/actual glucose values
    pred_values : array-like
        The predicted glucose values
    title_string : str
        The title of the plot
    
    Returns:
    --------
    Tuple[plt.Figure, List[int]]
        The figure with the Clarke Error Grid and zone counts [A, B, C, D, E]
    """
    ref_values = np.array(ref_values).flatten()
    pred_values = np.array(pred_values).flatten()

    assert (len(ref_values) == len(pred_values)), \
        f"Unequal number of values (reference: {len(ref_values)}) (prediction: {len(pred_values)})."

    if max(ref_values) > 400 or max(pred_values) > 400:
        logger.warning(f"Maximum reference value {max(ref_values)} or prediction value {max(pred_values)} exceeds 400 mg/dl.")
    if min(ref_values) < 0 or min(pred_values) < 0:
        logger.warning(f"Minimum reference value {min(ref_values)} or prediction value {min(pred_values)} is less than 0 mg/dl.")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(ref_values, pred_values, marker='o', color='blue', s=10, alpha=0.6)
    ax.set_title(title_string + " Clarke Error Grid", fontsize=16)
    ax.set_xlabel("True Glucose Value (mg/dl)", fontsize=14)
    ax.set_ylabel("Prediction Glucose Value (mg/dl)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_facecolor('white')
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_aspect('equal')

    # Draw zone boundaries
    ax.plot([0, 400], [0, 400], ':', c='black', linewidth=1)
    ax.plot([0, 175/3], [70, 70], '-', c='black', linewidth=1)
    ax.plot([175/3, 400/1.2], [70, 400], '-', c='black', linewidth=1)
    ax.plot([70, 70], [84, 400], '-', c='black', linewidth=1)
    ax.plot([0, 70], [180, 180], '-', c='black', linewidth=1)
    ax.plot([70, 290], [180, 400], '-', c='black', linewidth=1)
    ax.plot([70, 70], [0, 56], '-', c='black', linewidth=1)
    ax.plot([70, 400], [56, 320], '-', c='black', linewidth=1)
    ax.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
    ax.plot([180, 400], [70, 70], '-', c='black', linewidth=1)
    ax.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
    ax.plot([240, 400], [180, 180], '-', c='black', linewidth=1)
    ax.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

    # Add zone labels
    ax.text(30, 15, "A", fontsize=15, color='green', fontweight='bold')
    ax.text(370, 260, "B", fontsize=15, color='orange', fontweight='bold')
    ax.text(280, 370, "B", fontsize=15, color='orange', fontweight='bold')
    ax.text(160, 370, "C", fontsize=15, color='red', fontweight='bold')
    ax.text(160, 15, "C", fontsize=15, color='red', fontweight='bold')
    ax.text(30, 140, "D", fontsize=15, color='red', fontweight='bold')
    ax.text(370, 120, "D", fontsize=15, color='red', fontweight='bold')
    ax.text(30, 370, "E", fontsize=15, color='red', fontweight='bold')
    ax.text(370, 15, "E", fontsize=15, color='red', fontweight='bold')

    # Calculate zone assignments
    zone = [0] * 5
    for i in range(len(ref_values)):
        r = ref_values[i]
        p = pred_values[i]
        if (r <= 70 and p <= 70) or (p <= 1.2 * r and p >= 0.8 * r):
            zone[0] += 1  # Zone A
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            zone[4] += 1  # Zone E
        elif ((r >= 70 and r <= 290) and p >= r + 110) or ((r >= 130 and r <= 180) and (p <= (7 / 5) * r - 182)):
            zone[2] += 1  # Zone C
        elif (r >= 240 and (p >= 70 and p <= 180)) or (r <= 175 / 3 and p <= 180 and p >= 70) or ((r >= 175 / 3 and r <= 70) and p >= (6 / 5) * r):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    total = len(ref_values)
    zone_text = (
        f"Zone A: {zone[0]} points ({100*zone[0]/total:.1f}%)\n"
        f"Zone B: {zone[1]} points ({100*zone[1]/total:.1f}%)\n"
        f"Zone C: {zone[2]} points ({100*zone[2]/total:.1f}%)\n"
        f"Zone D: {zone[3]} points ({100*zone[3]/total:.1f}%)\n"
        f"Zone E: {zone[4]} points ({100*zone[4]/total:.1f}%)"
    )
    
    ax.text(0.5, 0.02, zone_text, transform=ax.transAxes, ha='center',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    return fig, zone


def evaluate_glucose_predictions_with_clarke(eval_data: Dict[str, Any], scaler_y) -> Dict[str, Any]:
    """
    Evaluate glucose predictions using Clarke Error Grid analysis.

    Parameters:
    -----------
    eval_data : dict
        Dictionary containing test data and predictions
    scaler_y : MinMaxScaler
        Scaler for glucose values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and zone statistics
    """
    from ..models.multi_horizon_model import convert_predictions_to_original_scale
    
    logger.info("Starting Clarke Error Grid analysis")
    
    y_test_orig = convert_predictions_to_original_scale(eval_data['y_test_60'], scaler_y)
    y_pred_60_orig = convert_predictions_to_original_scale(eval_data['y_pred_60'], scaler_y)
    y_pred_120_orig = convert_predictions_to_original_scale(eval_data['y_pred_120'], scaler_y)

    print("Generating Clarke Error Grid for 60-minute predictions...")
    fig_60, zone_60 = clarke_error_grid(y_test_orig, y_pred_60_orig, "60-minute Prediction")
    plt.savefig('clarke_error_grid_60min.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Generating Clarke Error Grid for 120-minute predictions...")
    fig_120, zone_120 = clarke_error_grid(y_test_orig, y_pred_120_orig, "120-minute Prediction")
    plt.savefig('clarke_error_grid_120min.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    plt.sca(axes[0])
    clarke_error_grid(y_test_orig, y_pred_60_orig, "60-minute Prediction")
    
    plt.sca(axes[1])
    clarke_error_grid(y_test_orig, y_pred_120_orig, "120-minute Prediction")
    
    plt.tight_layout()
    plt.savefig('combined_clarke_error_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate clinical acceptability
    total_60 = len(y_test_orig)
    total_120 = len(y_test_orig)
    clinical_acceptability_60 = 100 * (zone_60[0] + zone_60[1]) / total_60
    clinical_acceptability_120 = 100 * (zone_120[0] + zone_120[1]) / total_120

    # Print summary
    print("\nClarke Error Grid Analysis Summary:")
    print("====================================")
    print("60-minute Prediction Results:")
    for i, zone_name in enumerate(['A', 'B', 'C', 'D', 'E']):
        print(f"Zone {zone_name}: {zone_60[i]} points ({100*zone_60[i]/total_60:.1f}%)")
    
    print("\n120-minute Prediction Results:")
    for i, zone_name in enumerate(['A', 'B', 'C', 'D', 'E']):
        print(f"Zone {zone_120[i]} points ({100*zone_120[i]/total_120:.1f}%)")

    print(f"\nClinical Acceptability (Zones A+B):")
    print(f"60-minute Predictions: {clinical_acceptability_60:.1f}%")
    print(f"120-minute Predictions: {clinical_acceptability_120:.1f}%")

    results = {
        '60min': {
            'zones': zone_60,
            'percentages': [100*zone_60[i]/total_60 for i in range(5)],
            'clinical_acceptability': clinical_acceptability_60
        },
        '120min': {
            'zones': zone_120,
            'percentages': [100*zone_120[i]/total_120 for i in range(5)],
            'clinical_acceptability': clinical_acceptability_120
        }
    }

    logger.info("Clarke Error Grid analysis completed")
    return results