# Battery-SOH-Forecasting-Using-Temporal-Convolution-and-Channel-Attention
This project presents a deep learning framework for Battery State of Health (SOH) prediction using a Temporal Convolutional Network with Squeeze-and-Excitation. The proposed approach predicts future battery SOH using only historical SOH measurements, eliminating the need for dataset-specific feature engineering or additional input variables.
# Deep Learning-Based Battery State of Health Prediction Using TCN-SE

A deep learning framework for **Battery State of Health (SOH) prediction** using only historical SOH sequences. The proposed framework combines a **Temporal Convolutional Network (TCN)** with a **Squeeze-and-Excitation (SE)** attention mechanism and **Mish activation** to capture long-term battery degradation patterns with a relatively small computational footprint.

The framework is evaluated using a strict **Leave-One-Sequence-Out (LOSO)** validation strategy across three battery datasets with substantially different degradation characteristics: **NASA, CALCE, and Oxford**.

---

## Overview

Accurate SOH estimation is an important component of modern Battery Management Systems (BMS), particularly for electric vehicles and cloud-based battery monitoring.

Many existing deep learning approaches use multivariate measurements such as:

* Voltage
* Current
* Temperature
* Impedance
* Other battery-specific health indicators

Although these measurements can provide useful information, they can also introduce additional preprocessing, feature engineering, computational cost, environmental noise, and dataset-specific tuning.

This project investigates a different approach:

> **Can historical SOH values alone provide sufficient information for accurate future SOH forecasting?**

The proposed solution uses a **univariate TCN-SE architecture** that directly learns degradation patterns from historical SOH trajectories without handcrafted feature engineering.

The framework combines:

* Sliding-window time-series formulation
* Training-only Min-Max normalization
* Squeeze-and-Excitation channel recalibration
* Dilated causal convolutions
* Temporal Convolutional Network residual blocks
* Mish activation
* Huber loss
* Early stopping
* Strict Leave-One-Sequence-Out validation

The model achieved **R² values above 0.99 for the reported dataset-level averages** and maintained the same core hyperparameters across all three datasets.

---

## Key Features

* **Univariate input:** Uses only historical SOH values.
* **No handcrafted features:** The network directly learns temporal degradation representations.
* **TCN-based temporal modeling:** Dilated causal convolutions capture long-range dependencies.
* **SE attention:** Dynamically recalibrates learned feature channels.
* **Mish activation:** Provides smooth nonlinear activation for the TCN blocks.
* **Leakage-aware normalization:** Min-Max scaling is fitted exclusively using training data.
* **LOSO validation:** Each battery is completely held out during testing.
* **Fixed hyperparameters:** The same core configuration is used across NASA, CALCE, and Oxford.
* **Lightweight design:** Avoids the quadratic complexity associated with standard self-attention mechanisms.
* **Cross-dataset evaluation:** Tested on monotonic, regenerative, and dynamic degradation profiles.

---

# Dataset

Three publicly available benchmark datasets are considered.

## 1. NASA Battery Dataset

The NASA dataset represents an accelerated aging condition at approximately **24°C**.

Cells used:

* B0005
* B0006
* B0007
* B0018

The degradation trajectories are comparatively smooth and predominantly monotonic.

---

## 2. CALCE Battery Dataset

The CALCE dataset contains more complex degradation behavior, including **capacity regeneration and measurement noise**.

Cells used:

* CS2 35
* CS2 36
* CS2 37
* CS2 38

The presence of short-term capacity regeneration makes this dataset useful for testing whether the model can follow nonlinear fluctuations while preserving the long-term degradation trajectory.

---

## 3. Oxford Battery Dataset

The Oxford dataset represents batteries subjected to **dynamic Artemis electric vehicle drive cycles**.

Cells used:

* Cell 1
* Cell 3
* Cell 4
* Cell 7
* Cell 8

This dataset provides a more dynamic operating profile compared with the smoother NASA degradation trajectories.

---

# SOH Formulation

Battery capacity is obtained using Coulomb counting:

$$
C_t = \int_0^T I(t)\,dt
$$

The State of Health is then defined as:

$$
SOH_t = \frac{C_t}{C_0}\times100\%
$$

where:

* \(C_t\) = releasable capacity at cycle \(t\)
* \(C_0\) = nominal/initial capacity
* \(SOH_t\) = battery State of Health

The problem is formulated as a **univariate time-series forecasting problem**.

Given a historical sequence:

$$
X_t=[s_{t-L+1},...,s_t]
$$

the model predicts:

$$
\hat{s}_{t+1}=f(X_t;\theta)
$$

where \(L\) is the historical look-back window.

---

# Complete Workflow

```text
Battery Dataset
      │
      ▼
Extract Battery Capacity / SOH
      │
      ▼
Individual Battery SOH Sequences
      │
      ▼
Leave-One-Sequence-Out Split
      │
      ├───────────────┐
      │               │
 Training Batteries   Test Battery
      │
      ▼
Training-only Min-Max Scaling
      │
      ▼
Sliding Window Generation
      │
      │  L = 16
      ▼
Normalized SOH Sequence
      │
      ▼
Initial 1D Convolution
      │
      ▼
Squeeze-and-Excitation Block
      │
      ▼
TCN Residual Block
      │
      ├── Dilation = 1
      │
      ├── Dilation = 2
      │
      └── Dilation = 4
      │
      ▼
Final Time-Step Feature Vector
      │
      ▼
Dense Layer
      │
      ▼
Linear Regression Output
      │
      ▼
Predicted Next SOH
      │
      ▼
Inverse Scaling
      │
      ▼
RMSE / MAE / R²
```

The complete methodology uses training-exclusive normalization and a battery-level LOSO split to reduce the possibility of forward data leakage.

---

# Data Preprocessing

## 1. SOH Sequence Construction

Each battery is represented as a chronological SOH sequence.

No additional handcrafted statistical or physical features are required by the proposed model.

---

## 2. Sliding Window

A fixed historical window of:

```text
L = 16
```

is used.

For example:

```text
Input:
SOH[t-15], SOH[t-14], ..., SOH[t]

Target:
SOH[t+1]
```

This transforms the original SOH trajectory into supervised learning samples.

---

## 3. Training-Only Min-Max Scaling

The SOH sequences are normalized using:

$$
X_{norm} =
\frac{X-X_{min}}
{X_{max}-X_{min}}
$$

Importantly, the scaling parameters are obtained **only from the training data**.

The held-out battery is not used to calculate the normalization parameters.

This is an important part of the leakage-aware validation strategy.

---

# Model Architecture

The proposed architecture consists of three major stages:

```text
Input SOH Sequence
        │
        ▼
Feature Projection
        │
        ▼
Squeeze-and-Excitation
        │
        ▼
Temporal Convolutional Network
        │
        ▼
Global Integration Head
        │
        ▼
Predicted SOH
```

The architecture is designed to capture both **local temporal patterns** and **long-range degradation dependencies**.

---

# 1. Initial 1D Convolution

The input sequence is first projected into a higher-dimensional feature representation using a 1D convolutional layer.

Configuration:

```text
Filters      : 32
Kernel size : 3
```

This produces a multi-channel temporal feature representation that is subsequently processed by the SE block.

---

# 2. Squeeze-and-Excitation Block

The SE block provides adaptive channel-wise feature recalibration.

### Squeeze

Global Average Pooling is used to compress the temporal dimension into a channel descriptor:

$$
z_c=\frac{1}{L}\sum_{i=1}^{L}u_c(i)
$$

### Excitation

A gating mechanism then learns channel importance:

$$
s=\sigma(W_2\delta(W_1z))
$$

where:

* \(\delta\) = ReLU
* \(\sigma\) = Sigmoid
* Reduction ratio \(r=8\)

The recalibrated feature map is:

$$
\tilde{X}_c=s_c\cdot u_c
$$

This allows the model to assign different importance to learned temporal feature channels.

---

# 3. Temporal Convolutional Network

The TCN backbone is responsible for learning long-term temporal dependencies.

Instead of recurrent processing, the architecture uses **dilated causal convolutions**.

The dilation factors are:

```text
d = 1
d = 2
d = 4
```

with:

```text
Kernel size = 5
```

Three TCN residual blocks are stacked.

The dilation mechanism increases the receptive field without requiring a very deep network, allowing the model to observe longer portions of the degradation history.

The causal structure also preserves temporal ordering because future observations cannot influence earlier predictions.

---

# 4. Mish Activation

The TCN blocks use **Mish** instead of the conventional ReLU activation:

$$
f(x)=x\tanh(\ln(1+e^x))
$$

Mish provides a continuously differentiable nonlinear activation and was investigated specifically for improving gradient propagation during nonlinear battery capacity fading.

The ablation study also evaluates the effect of replacing Mish with standard ReLU.

---

# 5. Global Integration Head

Instead of flattening the complete temporal feature map, the framework extracts the feature vector from the **final time step**.

The resulting representation has:

```text
64 × 1
```

dimensions.

It is then passed through:

```text
Dense layer : 32 neurons
Activation  : Mish

Output layer: 1 neuron
Activation  : Linear
```

The final neuron produces the predicted next SOH value.

---

# Training Configuration

The same core hyperparameters are maintained across all three datasets.

| Parameter            | Configuration |
| -------------------- | ------------- |
| Look-back window     | 16            |
| Batch size           | 16            |
| Optimizer            | Adam          |
| Learning rate        | 0.001         |
| Loss function        | Huber Loss    |
| TCN kernel size      | 5             |
| TCN dilation factors | 1, 2, 4       |
| Initial Conv filters | 32            |
| SE reduction ratio   | 8             |
| Dense layer          | 32 neurons    |
| TCN activation       | Mish          |

Huber Loss is used because it is less sensitive to outlier capacity-regeneration spikes, which is particularly relevant for the CALCE dataset.

---

# Validation Strategy

## Leave-One-Sequence-Out (LOSO)

The evaluation is performed at the **battery level**, rather than randomly splitting individual time-series windows.

For a dataset containing \(K\) batteries:

```text
Training:
K - 1 batteries

Testing:
1 completely unseen battery
```

This process is repeated until every battery has been used as the held-out test sequence.

During training, 20% of the training data is used for validation, with early stopping applied when validation loss stops improving.

### Why LOSO?

Randomly splitting overlapping time-series windows could allow highly similar portions of the same degradation trajectory to appear in both training and testing.

LOSO instead evaluates whether the model can generalize to an **unseen battery degradation trajectory**.

---

# Evaluation Metrics

Three metrics are used.

### RMSE

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
}
$$

RMSE penalizes larger prediction errors more strongly.

### MAE

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

MAE measures the average absolute prediction error.

### R²

$$
R^2 =
1-
\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
$$

Higher R² indicates that a larger proportion of the variance in the actual SOH trajectory is explained by the predictions.

---



<img width="1515" height="409" alt="image" src="https://github.com/user-attachments/assets/15333677-d7b1-4dd8-b548-b168c82113e4" />
                                          Figure 1 — Dataset Capacity Degradation Trajectories


# TCN-SE Architecture

```markdown
Input SOH Sequence
       │
       ▼
1D Convolution (Feature Projection)
       │
  ┌────┴──────────────────────────┐
  │                               ▼
  │               Squeeze-and-Excitation (SE) Block
  │             ┌───────────────────────────────────┐
  │             │   Global Average Pooling 1D       │
  │             │                 │                 │
  │             │                 ▼                 │
  │             │            Dense + ReLU           │
  │             │                 │                 │
  │             │                 ▼                 │
  │             │           Dense + Sigmoid         │
  │             └─────────────────┬─────────────────┘
  │                               │ (Attention Weights)
  │                               ▼
  └───────────────► Scale (Channel-wise Mult) ◄── (Skip Connection)
                                  │
                                  ▼
                Temporal Convolutional Network (TCN)
                ┌───────────────────────────────────┐
                │   TCN Block 1 (Dilation: 1)       │
                │                 │                 │
                │                 ▼                 │
                │   TCN Block 2 (Dilation: 2)       │
                │                 │                 │
                │                 ▼                 │
                │   TCN Block 3 (Dilation: 4)       │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                       Global Integration Head
                ┌───────────────────────────────────┐
                │      Extract Last Time-step       │
                │                 │                 │
                │                 ▼                 │
                │     Fully Connected (Dense 32)    │
                │                 │                 │
                │                 ▼                 │
                │     Fully Connected (Dense 1)     │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
               Predicted SOH (Output Forecast at t+1)
```


<img width="961" height="694" alt="image" src="https://github.com/user-attachments/assets/4d5acd4c-bdd9-4b2d-9d57-cb66dfc08a7e" />
                                             Figure 3 — Actual vs Predicted SOH

The paper reports that Figure 3 contains the actual and predicted SOH trajectories for all 13 evaluated battery cells.

---

# Results

## Overall Performance

The model was evaluated across:

```text
NASA   : 4 cells
CALCE  : 4 cells
Oxford : 5 cells

Total  : 13 battery cells
```

### Dataset-Level Results

| Dataset | Average RMSE | Average MAE | Average R² |
| ------- | -----------: | ----------: | ---------: |
| NASA    |       0.0049 |      0.0035 |     0.9963 |
| CALCE   |       0.0128 |      0.0079 |     0.9958 |
| Oxford  |       0.0024 |      0.0019 |     0.9951 |

The Oxford dataset produced the lowest average RMSE and MAE, while all three datasets achieved average R² values above 0.995.

---

## NASA Results

| Cell        |            RMSE |        MAE |         R² |
| ----------- | --------------: | ---------: | ---------: |
| B0005       | 0.0025 ± 0.0006 |     0.0020 |     0.9993 |
| B0006       | 0.0109 ± 0.0026 |     0.0074 |     0.9894 |
| B0007       | 0.0029 ± 0.0011 |     0.0023 |     0.9985 |
| B0018       | 0.0031 ± 0.0003 |     0.0021 |     0.9980 |
| **Average** |      **0.0049** | **0.0035** | **0.9963** |

The strongest NASA result was obtained for B0005, with:

```text
RMSE = 0.0025
MAE  = 0.0020
R²   = 0.9993
```

The model maintained R² values above 0.998 for B0005, B0007, and B0018.

---

## CALCE Results

| Cell        |            RMSE |        MAE |         R² |
| ----------- | --------------: | ---------: | ---------: |
| CS2 35      | 0.0118 ± 0.0013 |     0.0074 |     0.9960 |
| CS2 36      | 0.0151 ± 0.0011 |     0.0087 |     0.9958 |
| CS2 37      | 0.0126 ± 0.0021 |     0.0083 |     0.9954 |
| CS2 38      | 0.0118 ± 0.0009 |     0.0071 |     0.9959 |
| **Average** |      **0.0128** | **0.0079** | **0.9958** |

Despite the nonlinear capacity regeneration and increased measurement noise in the CALCE sequences, the model maintained R² values above 0.995 for all four evaluated cells.

---

## Oxford Results

| Cell        |            RMSE |        MAE |         R² |
| ----------- | --------------: | ---------: | ---------: |
| Cell 1      | 0.0019 ± 0.0002 |     0.0016 |     0.9982 |
| Cell 3      | 0.0019 ± 0.0003 |     0.0014 |     0.9983 |
| Cell 4      | 0.0039 ± 0.0006 |     0.0027 |     0.9844 |
| Cell 7      | 0.0023 ± 0.0003 |     0.0019 |     0.9967 |
| Cell 8      | 0.0021 ± 0.0004 |     0.0017 |     0.9981 |
| **Average** |      **0.0024** | **0.0019** | **0.9951** |

The model achieved particularly low errors on Cells 1 and 3, while Cell 4 represented a more difficult prediction case.

---

# Ablation Study

To investigate the contribution of individual architectural components, three variants were compared against the complete model:

1. Baseline CNN without TCN and SE
2. TCN-SE using ReLU instead of Mish
3. TCN with Mish but without the SE block
4. Proposed TCN-SE with Mish

| Model                      |        RMSE |         MAE |         R² |
| -------------------------- | ----------: | ----------: | ---------: |
| Baseline CNN               |     0.05323 |     0.04753 |    -2.3690 |
| Standard ReLU (TCN-SE)     |     0.05097 |     0.04538 |    -2.0880 |
| No SE Block (TCN + Mish)   |     0.00923 |     0.00781 |     0.8988 |
| **Proposed TCN-SE + Mish** | **0.00713** | **0.00617** | **0.9396** |

The ablation results indicate that:

* The baseline CNN struggles to model long-term temporal dependencies.
* Replacing Mish with ReLU substantially degrades performance in the tested ablation setting.
* Removing the SE block reduces performance while retaining a functional TCN backbone.
* The complete **TCN + SE + Mish** configuration provides the strongest result among the tested variants.

The reported ablation study therefore supports the inclusion of both channel recalibration and Mish activation in the proposed architecture.

---

# Comparison With Existing Deep Learning Approaches

## NASA

For B0005:

| Model               |       RMSE |        MAE |         R² |
| ------------------- | ---------: | ---------: | ---------: |
| LSTM                |     0.0286 |     0.0246 |     0.9805 |
| Standard TCN        |     0.0180 |     0.0140 |          — |
| PSO-LSTM-Attention  |     0.0163 |          — |          — |
| GRU + MHA + Ridge   |     0.0070 |     0.0050 |     0.9964 |
| **Proposed TCN-SE** | **0.0025** | **0.0020** | **0.9993** |

The proposed model also reports lower errors than the listed comparison approaches for B0007 and B0018.

---

## CALCE

| Cell   | Proposed RMSE | Proposed MAE | Proposed R² |
| ------ | ------------: | -----------: | ----------: |
| CS2 35 |        0.0118 |       0.0074 |      0.9960 |
| CS2 36 |        0.0151 |       0.0087 |      0.9958 |
| CS2 37 |        0.0126 |       0.0083 |      0.9954 |
| CS2 38 |        0.0118 |       0.0071 |      0.9959 |

The reported comparisons include LSTM, Transformer, PatchTST, and TCN-t Transformer models. The proposed framework maintains competitive performance while using a univariate input formulation.

---

## Oxford

| Cell   | Proposed RMSE | Proposed MAE | Proposed R² |
| ------ | ------------: | -----------: | ----------: |
| Cell 1 |        0.0019 |       0.0016 |      0.9982 |
| Cell 3 |        0.0019 |       0.0014 |      0.9983 |
| Cell 4 |        0.0039 |       0.0027 |      0.9844 |
| Cell 7 |        0.0023 |       0.0019 |      0.9967 |
| Cell 8 |        0.0021 |       0.0017 |      0.9981 |

For Cell 7, the proposed model reports an RMSE of 0.0023 compared with 0.0101 for the listed LSTM baseline.

---

# Why TCN-SE?

The architecture combines three complementary ideas.

### Temporal Convolutional Network

Dilated causal convolutions provide an expanding receptive field, allowing the model to capture degradation dependencies over longer temporal horizons.

### Squeeze-and-Excitation

The SE mechanism learns channel-wise importance and recalibrates intermediate features according to their relevance.

### Mish

Mish provides a smooth nonlinear activation and was selected to improve gradient propagation during nonlinear capacity degradation.

Together, these components allow the model to learn long-term degradation patterns while retaining a relatively compact architecture.

---

# Computational Perspective

The framework intentionally avoids multivariate sensor streams and large self-attention architectures.

The paper describes the proposed TCN-based framework as having **linear sequence complexity, O(L)**, in contrast to the quadratic sequence scaling associated with standard self-attention, **O(L²)**.

This design is motivated by potential deployment in resource-constrained or cloud-based BMS environments.

---

# Reproducibility Workflow

A reproducible implementation can follow the sequence below:

```text
1. Load NASA / CALCE / Oxford battery data

2. Extract capacity measurements

3. Calculate SOH for each battery

4. Keep each battery as an independent chronological sequence

5. Create LOSO train/test folds

6. Fit Min-Max scaler using training batteries only

7. Normalize training and held-out test sequences

8. Generate sliding windows
      Look-back = 16

9. Build TCN-SE model

10. Train using:
      Adam
      Learning rate = 0.001
      Huber Loss
      Batch size = 16

11. Use 20% training data for validation

12. Apply early stopping

13. Predict the held-out battery

14. Inverse-transform predictions

15. Calculate:
      RMSE
      MAE
      R²

16. Repeat for every battery

17. Average results across cells

18. Generate actual-vs-predicted SOH plots

19. Perform ablation experiments

20. Compare against existing architectures
```

---

# Suggested Repository Structure

```text
TCN-SE-Battery-SOH/
│
├── README.md
│
├── data/
│   ├── NASA/
│   ├── CALCE/
│   └── Oxford/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_soh_generation.ipynb
│   ├── 03_tcn_se_training.ipynb
│   ├── 04_loso_evaluation.ipynb
│   └── 05_ablation_study.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sequence_generation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
├── figures/
│   ├── fig1_dataset_degradation.png
│   ├── fig2_tcn_se_architecture.png
│   └── fig3_soh_forecasting.png
│
├── requirements.txt
└── LICENSE
```

---

# Technologies

The implementation is based on the following machine learning concepts and tools:

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Deep Learning
* Time-Series Forecasting
* Temporal Convolutional Networks
* Squeeze-and-Excitation Networks

---

# Main Contributions

The project investigates a lightweight alternative to multivariate battery SOH prediction by:

1. Formulating SOH prediction as a **univariate sequence forecasting problem**.
2. Combining **TCN and SE attention** for temporal feature extraction.
3. Using **Mish activation** for nonlinear trajectory modeling.
4. Applying **training-exclusive normalization** to reduce forward data leakage.
5. Using **battery-level LOSO validation** to evaluate generalization to unseen degradation trajectories.
6. Maintaining a **constant core hyperparameter configuration** across NASA, CALCE, and Oxford datasets.
7. Evaluating the architecture through both **ablation experiments and comparisons with existing deep learning approaches**.

---

# Limitations and Future Work

The current framework focuses on historical SOH sequences and therefore does not explicitly use physical measurements such as voltage, current, temperature, or impedance.

Future work described in the study includes:

* Evaluation across additional battery chemistries, including solid-state batteries.
* Validation on broader operating conditions.
* End-to-end integration with IoT-based battery monitoring hardware.
* Further investigation of deployment in real-world Cloud BMS environments.

---

# Citation

If you use this implementation or methodology in academic work, please cite the corresponding paper:

```bibtex
@article{tcn_se_battery_soh,
  title   = {A Lightweight Univariate TCN-SE Framework for Battery State of Health Prediction},
  author  = {Author Names},
  journal = {Under Review},
  year    = {2026}
}
```

> Replace the citation information with the final bibliographic details after publication.

---

# Results at a Glance

```text
Datasets evaluated       : NASA, CALCE, Oxford
Battery cells             : 13
Input                     : Historical SOH only
Look-back window          : 16
Architecture              : TCN + SE + Mish
Validation                : Leave-One-Sequence-Out
Normalization             : Training-only Min-Max scaling
Optimizer                 : Adam
Loss                      : Huber
Best reported RMSE        : 0.0019
Best reported MAE         : 0.0014
Dataset-average R²        : > 0.995 across all three datasets
```

The reported results demonstrate that the proposed univariate TCN-SE framework can track diverse battery degradation trajectories while maintaining a consistent core configuration across the evaluated datasets.
