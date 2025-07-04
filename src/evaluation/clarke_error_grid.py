import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def clarke_error_grid(ref_values: np.ndarray, pred_values: np.ndarray, 
                     title_string: str = "Clarke Error Grid") -> Tuple[plt.Figure, List[int]]:
    """Generates a Clarke Error Grid for diabetes glucose predictions
    
    Parameters:
    -----------
    ref_values : array-like
        The reference/actual glucose values
    pred_values : array-like
        The predicted glucose values
    title_string : string
        The title of the plot
    
    Returns:
    --------
    plt : matplotlib figure
        The figure with the Clarke Error Grid
    zone : list
        A list with the number of points in each zone [A, B, C, D, E]
    """
    ref_values = np.array(ref_values).flatten()
    pred_values = np.array(pred_values).flatten()

    assert (len(ref_values) == len(pred_values)), \
        "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    if max(ref_values) > 400 or max(pred_values) > 400:
        print("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds 400 mg/dl.".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values), min(pred_values)))

    plt.figure(figsize=(10, 10))
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=10)
    plt.title(title_string + " Clarke Error Grid", fontsize=16)
    plt.xlabel("True Glucose Value (mg/dl)", fontsize=14)
    plt.ylabel("Prediction Glucose Value (mg/dl)", fontsize=14)
    plt.xticks(np.arange(0, 401, 50), fontsize=12)
    plt.yticks(np.arange(0, 401, 50), fontsize=12)
    plt.gca().set_facecolor('white')
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.gca().set_aspect('equal')

    plt.plot([0, 400], [0, 400], ':', c='black')
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    plt.text(30, 15, "A", fontsize=15, color='green', fontweight='bold')
    plt.text(370, 260, "B", fontsize=15, color='orange', fontweight='bold')
    plt.text(280, 370, "B", fontsize=15, color='orange', fontweight='bold')
    plt.text(160, 370, "C", fontsize=15, color='red', fontweight='bold')
    plt.text(160, 15, "C", fontsize=15, color='red', fontweight='bold')
    plt.text(30, 140, "D", fontsize=15, color='red', fontweight='bold')
    plt.text(370, 120, "D", fontsize=15, color='red', fontweight='bold')
    plt.text(30, 370, "E", fontsize=15, color='red', fontweight='bold')
    plt.text(370, 15, "E", fontsize=15, color='red', fontweight='bold')

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
    plt.figtext(0.5, 0.05,
                f"Zone A: {zone[0]} points ({100*zone[0]/total:.1f}%)\n"
                f"Zone B: {zone[1]} points ({100*zone[1]/total:.1f}%)\n"
                f"Zone C: {zone[2]} points ({100*zone[2]/total:.1f}%)\n"
                f"Zone D: {zone[3]} points ({100*zone[3]/total:.1f}%)\n"
                f"Zone E: {zone[4]} points ({100*zone[4]/total:.1f}%)",
                ha='center', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.grid(True, linestyle='--', alpha=0.3)

    return plt, zone

    

def evaluate_glucose_predictions_with_clarke(eval_data: dict, scaler_y) -> dict:
    """Evaluate glucose predictions using Clarke Error Grid analysis.

    Parameters:
    -----------
    eval_data : dict
        Dictionary containing test data and predictions
    scaler_y : MinMaxScaler
        Scaler for glucose values
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics and zone statistics
    """
    def convert_predictions_to_original_scale(y_pred, scaler_y):
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        return scaler_y.inverse_transform(y_pred).flatten()

    y_test_orig = convert_predictions_to_original_scale(eval_data['y_test_60'], scaler_y)
    y_pred_60_orig = convert_predictions_to_original_scale(eval_data['y_pred_60'], scaler_y)
    y_pred_120_orig = convert_predictions_to_original_scale(eval_data['y_pred_120'], scaler_y)

    print("Generating Clarke Error Grid for 60-minute predictions...")
    plt_60, zone_60 = clarke_error_grid(y_test_orig, y_pred_60_orig, "60-minute Prediction")
    plt.tight_layout()
    plt.savefig('clarke_error_grid_60min.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Generating Clarke Error Grid for 120-minute predictions...")
    plt_120, zone_120 = clarke_error_grid(y_test_orig, y_pred_120_orig, "120-minute Prediction")
    plt.tight_layout()
    plt.savefig('clarke_error_grid_120min.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.sca(axes[0])
    clarke_error_grid(y_test_orig, y_pred_60_orig, "60-minute Prediction")
    plt.sca(axes[1])
    clarke_error_grid(y_test_orig, y_pred_120_orig, "120-minute Prediction")
    plt.tight_layout()
    plt.savefig('combined_clarke_error_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

    total_60 = len(y_test_orig)
    total_120 = len(y_test_orig)

    print("\nClarke Error Grid Analysis Summary:")
    print("====================================")
    print("60-minute Prediction Results:")
    print(f"Zone A (Clinically Accurate): {zone_60[0]} points ({100*zone_60[0]/total_60:.1f}%)")
    print(f"Zone B (Benign Errors): {zone_60[1]} points ({100*zone_60[1]/total_60:.1f}%)")
    print(f"Zone C (Overcorrection): {zone_60[2]} points ({100*zone_60[2]/total_60:.1f}%)")
    print(f"Zone D (Dangerous Failure to Detect): {zone_60[3]} points ({100*zone_60[3]/total_60:.1f}%)")
    print(f"Zone E (Erroneous Treatment): {zone_60[4]} points ({100*zone_60[4]/total_60:.1f}%)")
    print("\n120-minute Prediction Results:")
    print(f"Zone A (Clinically Accurate): {zone_120[0]} points ({100*zone_120[0]/total_120:.1f}%)")
    print(f"Zone B (Benign Errors): {zone_120[1]} points ({100*zone_120[1]/total_120:.1f}%)")
    print(f"Zone C (Overcorrection): {zone_120[2]} points ({100*zone_120[2]/total_120:.1f}%)")
    print(f"Zone D (Dangerous Failure to Detect): {zone_120[3]} points ({100*zone_120[3]/total_120:.1f}%)")
    print(f"Zone E (Erroneous Treatment): {zone_120[4]} points ({100*zone_120[4]/total_120:.1f}%)")

    clinical_acceptability_60 = 100 * (zone_60[0] + zone_60[1]) / total_60
    clinical_acceptability_120 = 100 * (zone_120[0] + zone_120[1]) / total_120

    print("\nClinical Acceptability (Zones A+B):")
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

    return results
