import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformer import MLP_class, get_dataloader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import MulensModel as mm

# --- Configuration ---
MODEL_PATH = "chkpt.pth"
DATA_DIR = "data/generated_lightcurves/xy" # Base directory for the lightcurve files
PARAM_FILE = "data/generated_lightcurves/params.csv" # Full parameter file for all data
TEST_SPLIT_FILE = "data/generated_lightcurves/test.csv" # CSV file containing list of files for the test set
BATCH_SIZE = 32
OUTPUT_DIR = "figures" # Directory for saving plots

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# function to get smooth curve
def get_curve(args):
    my_1S2L_model = mm.Model({'t_0': 0, 'u_0': args[0],
                              't_E': args[1], 'rho': args[2], 'q': args[3], 's': args[4],
                              'alpha': args[5]})
    times = my_1S2L_model.set_times()
    times -= min(times)
    lc = my_1S2L_model.get_lc(source_flux=1)
    return times, lc

# Function to run the evaluation
def run_evaluation():
    print(f"Loading model from: {MODEL_PATH}")
    try:
        # Load the model structure and weights (assuming MLP_class is defined correctly)
        model = torch.load(MODEL_PATH, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {MODEL_PATH}. Run the training script first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set model to evaluation mode (crucial for disabling dropout/batchnorm)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_dl = get_dataloader(
        xy_dir=DATA_DIR,
        param_file=PARAM_FILE,
        batch_size=BATCH_SIZE,
        shuffle=False # Do not shuffle test data
    )

    all_predictions = []
    all_true_params = []
    
    # Store one example batch for visualization
    example_batch = None
    transformer_model = False
    with torch.no_grad(): # Disable gradient calculations during evaluation
        for step, (seqs, mask, params) in enumerate(test_dl):
            seqs = seqs.cuda()
            mask = mask.cuda()
            
            # Forward pass
            if transformer_model:
                pred = model(seqs, mask)
            else:
                pred = model(seqs)

            # Move data from GPU back to CPU and convert to numpy
            predictions_np = pred.cpu().numpy()
            params_np = params.numpy() # True parameters

            all_predictions.append(predictions_np)
            all_true_params.append(params_np)
            
            # Save the first batch for example visualization
            if example_batch is None:
                seqs_np = seqs.cpu().numpy()
                example_batch = (predictions_np, params_np, seqs_np)
                print(f"Example prediction generated (Batch 0). True shape: {params_np.shape}, Pred shape: {predictions_np.shape}, Seqs shape: {seqs_np.shape}")


    # Combine all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_params = np.concatenate(all_true_params, axis=0)
    
    # --- Metric Reporting ---
    
    print("\n--- Overall Performance Metrics ---")
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(all_true_params, all_predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = calculate_rmse(all_true_params, all_predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    if example_batch is not None:
        pred_ex, true_ex, seqs = example_batch
        
        pred_args = pred_ex[0]
        true_args = true_ex[0]
        arg_names = ['u_0', 't_E', 'rho', 'q', 's', 'alpha']
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(np.arange(6), true_args, 'o-', label='True Parameter', alpha=0.7)
        ax1.plot(np.arange(6), pred_args, 'x--', label='Predicted Parameter', alpha=0.7)
        ax1.set_xticks(np.arange(6))
        ax1.set_xticklabels(arg_names)
        ax1.set_ylabel("Value")
        ax1.legend()

        x = seqs[0,:,0]
        y = seqs[0,:,1]
        ax2.plot(x, y, label="Noisy Data")
        x_fit, y_fit = get_curve(pred_args.tolist())
        y_fit = np.interp(x, x_fit, y_fit)
        ax2.plot(x, y_fit, label="Fitted Curve")
        ax2.legend()
        ax2.set_ylabel("Magnification Factor")
        ax2.set_xlabel("Time")
        plt.tight_layout()
        plot_filename = os.path.join(OUTPUT_DIR, 'example_predictions.png')
        plt.savefig(plot_filename)
        print(f"Example prediction plot saved to {plot_filename}")

if __name__ == "__main__":
    run_evaluation()