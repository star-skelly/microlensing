import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformer import MLP_class, get_dataloader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

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
                example_batch = (predictions_np, params_np)
                print(f"Example prediction generated (Batch 0). True shape: {params_np.shape}, Pred shape: {predictions_np.shape}")


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
    
    # --- Example Visualization (Predicted vs. True) ---

    if example_batch is not None:
        pred_ex, true_ex = example_batch
        
        # Determine the number of predicted parameters (dimensionality of the output)
        num_params = pred_ex.shape[1] 
        
        # Create a subplot for each parameter
        fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 5))
        
        # Handle case where there is only 1 parameter (axes is not an array)
        if num_params == 1:
            axes = [axes]
        
        print(f"\nVisualizing example predictions for {num_params} parameters...")

        for i in range(num_params):
            ax = axes[i]
            # Plot the predictions and true values for this specific parameter across the batch
            # We use indices (0 to BATCH_SIZE-1) on the x-axis to represent different samples in the batch
            sample_indices = np.arange(len(pred_ex))
            
            ax.plot(sample_indices, true_ex[:, i], 'o-', label='True Parameter', alpha=0.7)
            ax.plot(sample_indices, pred_ex[:, i], 'x--', label='Predicted Parameter', alpha=0.7)
            
            # Calculate the individual parameter's MAE and RMSE for the overall test set
            param_mae = mean_absolute_error(all_true_params[:, i], all_predictions[:, i])
            param_rmse = calculate_rmse(all_true_params[:, i], all_predictions[:, i])

            ax.set_title(f'Parameter {i+1} Prediction Example\nMAE: {param_mae:.3f}, RMSE: {param_rmse:.3f}')
            ax.set_xlabel('Sample Index in Batch')
            ax.set_ylabel(f'Parameter Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_filename = os.path.join(OUTPUT_DIR, 'example_predictions.png')
        plt.savefig(plot_filename)
        print(f"Example prediction plot saved to {plot_filename}")

if __name__ == "__main__":
    run_evaluation()