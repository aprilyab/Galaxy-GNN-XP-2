# Report
=====================================

This report compares the performance of the Galaxy Tool Prediction LSTM model across three key stages of development:
1.  **Baseline**: Initial model before hyperparameter tuning.
2.  **First Tuning (High Capacity)**: 128/256 Bi-LSTM with moderate regularization (Dropout 0.3).
3.  **Second Tuning (Optimized)**: 64/128 Uni-LSTM with aggressive regularization (Dropout 0.5) and Ranking Loss.

## Performance Evolution
| Metric      | Baseline (Before Tuning) | First Tuning (High Capacity) | Second Tuning (Optimized) |
| :---        | :---:                    | :---:                        | :---:                     |
| **Precision** | 0.650                  | 0.750                        | 0.500                     |
| **Recall**    | 0.650                  | 0.750                        | 0.500                     |
| **Hit@5**     | 0.700                  | **0.900**                    | 0.750                     |
| **MRR**       | 0.677                  | **0.831**                    | 0.637                     |

### Analysis
- **First Tuning (Best Performance)**: The High Capacity model (Bi-LSTM, 256 hidden units) achieved the best results across all metrics (Hit@5: 90%).
- **Second Tuning**: Reducing model size and increasing regularization led to underfitting.

## Training Convergence & Configuration
#  Hyperparameter Configuration

##config = {
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-3,
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "patience": 7,
    "grad_clip": 1.0,
    "label_smoothing": 0.1
}

### 1. Baseline (Before Tuning)
![Baseline Loss](/home/henok/Desktop/Galaxy-GNN-XP-2/reports/Traning_loss_before_HP_Tunning.png)

### 2. First Tuning (High Capacity) - 
- **Architecture**: Bi-directional LSTM (Embed=128, Hidden=256)
- **Regularization**: Dropout=0.3
- **Loss**: CrossEntropy

![First Tuning Loss](/home/henok/Desktop/Galaxy-GNN-XP-2/reports/traing%20vs%20validation_loss_after_HR_tunning.png)

### 3. Second Tuning (Optimized)
- **Architecture**: Unidirectional LSTM (Embed=64, Hidden=128)
- **Regularization**: Dropout=0.5, Ranking Loss

![Second Tuning Loss](/home/henok/Desktop/Galaxy-GNN-XP-2/reports/loss_convergence_report.png)

## Model Paths
- **Best Model (First Tuning)**: `/home/henok/Desktop/Galaxy-GNN-XP-2/Outputs/best_galaxy_lstm_20260213_221606.pth`
- **Optimized Model (Second Tuning)**: `/home/henok/Desktop/Galaxy-GNN-XP-2/Outputs/best_galaxy_lstm_opt_20260214_100048.pth`
