# Lightweight-Audio-Distillation
Lightweight audio classification framework achieving state-of-the-art or highly competitive results across multiple benchmarks while using dramatically fewer parameters and computational resources, enabling accurate and efficient deployment on resource-constrained and edge devices.
%%%
# Lightweight Audio Classification with Staged Knowledge Distillation

A lightweight audio classification framework designed to push accuracy as high as possible while keeping **parameter count, memory footprint, and computation extremely small**.

The core idea is to train a compact CNN through **structurally aligned feature supervision, temperature-scaled logit distillation, and staged teacher guidance**, then discard every teacher at inference time.

The result is a tiny deployable model that reaches:

| Dataset                       |   Accuracy | Parameters |       Model Size |
| ----------------------------- | ---------: | ---------: | ---------------: |
| **TAU Urban Acoustic Scenes** | **60.24%** |  **31.2K** | **62.5 KB FP16** |
| **ESC-50**                    |  **93.4%** |  **36.4K** | **72.8 KB FP16** |
| **UrbanSound8K**              | **94.43%** |  **31.2K** | **62.5 KB FP16** |

For TAU, the model achieves the strongest result among the compared low-complexity systems while using only **31.2K parameters**, roughly half the parameter count of the smallest competing system.

On the DCASE 2025 official evaluation setup, the system reaches **62.3% overall accuracy**, outperforming the first-ranked system by **0.7 percentage points** while using approximately half as many parameters.

---

## Highlights

* **31.2K parameters** for the 10-class compact model
* **62.5 KB** model size in FP16
* **28.6 MMACs** for a one-second TAU input
* **60.24%** accuracy on TAU
* **93.4%** accuracy on ESC-50
* **94.43%** accuracy on UrbanSound8K
* **62.3%** on the DCASE 2025 official evaluation setup
* No additional inference-time teacher network
* No projection layers required for feature matching
* Intermediate feature supervision
* Temperature-scaled logit distillation
* Multi-teacher ensemble supervision
* Huber-based normalized feature matching
* Device-robust augmentation
* FP16 deployment with negligible accuracy degradation
* Scales from roughly **31K parameters to 350K parameters**

The training pipeline is intentionally more expensive than inference. Once training is complete, all teacher networks are removed and only the compact student is deployed.

---

# Architecture

The system uses two structurally related networks:

1. a larger **aligned teacher**
2. a compact **student**

The two models preserve the same overall stage organization and compatible intermediate feature dimensions.

This makes it possible to directly supervise intermediate student representations without adding projection modules.

The compact student contains roughly **77% fewer parameters** than the aligned teacher in the 10-class configuration.

## Architecture Diagram

[Open the architecture diagram](images/Student - New.pdf)

Source path:

```text
images/Student - New.pdf
```

The backbone contains:

* a two-layer convolutional stem
* Light Echo blocks
* pointwise convolutions
* depthwise convolutions
* residual connections
* Residual Normalization
* teacher-student aligned convolution positions
* attention-based global pooling
* batch normalization
* dropout
* a final linear classifier

---

# Light Echo Block

The lightweight backbone uses a pointwise-depthwise residual design.

A Light Echo block combines:

```text
Input
  │
  ├── Pointwise Conv
  │
  ├── LeakyReLU
  │
  ├── Depthwise 3×3 Conv
  │
  ├── BatchNorm
  │
  ├── LeakyReLU
  │
  ├── Pointwise Conv
  │
  └── Residual / Projection Connection
```

This provides a favorable balance between representational capacity and computational cost.

---

# Structurally Aligned Teacher

Instead of using a completely unrelated teacher architecture, the larger teacher is deliberately designed to remain structurally compatible with the student.

Nine student `3×3` convolution positions are replaced by **FusionConv** blocks in the aligned teacher.

Conceptually:

```text
Aligned Teacher                 Student
---------------                 -------
FusionConv        <-------->     Conv 3×3
FusionConv        <-------->     Conv 3×3
FusionConv        <-------->     Conv 3×3
     ...                            ...
```

The stage layouts and feature dimensions remain compatible.

This allows direct feature matching:

```text
Teacher Feature ─────┐
                     ├── Feature Loss
Student Feature ─────┘
```

without inserting extra learned projection layers.

---

# FusionConv

FusionConv increases the capacity of the aligned teacher while preserving compatible output dimensions.

The block uses multiple convolutional branches:

```text
                    ┌── 3×3 Conv ── BN ──┐
                    │                     │
Input ──────────────┼── 5×5 Conv ── BN ──┤
                    │                     ├── Concat ── 1×1 Conv ── Output
                    ├── 7×7 Conv ── BN ──┤
                    │                     │
                    └── 3×1 Conv ── BN ──┘
```

This gives the teacher a richer receptive-field mixture while allowing its output representation to stay aligned with the student.

---

# Staged Knowledge Transfer

Training is split into three phases rather than optimizing every supervision signal simultaneously.

```text
Phase I
Aligned Teacher Training
        │
        ▼
Phase II
Student + Labels
        +
Aligned Teacher Logits
        +
Normalized Feature Matching
        │
        ▼
Phase III
Student + Labels
        +
External Teacher / Ensemble Logits
        │
        ▼
Compact Final Student
```

The student is gradually moved from:

**standalone learning → representation matching → stronger output-level guidance**

rather than receiving all supervision sources from the beginning.

## Full Pipeline Diagram

[Open the complete distillation pipeline](images/StPlusKnwldge_figure3.pdf)

Source path:

```text
images/StPlusKnwldge_figure3.pdf
```

---

# Distillation Formulation

Let

* $\mathbf{z}_s$ be the student logits
* $\mathbf{z}_i$ be the aligned-teacher logits
* $\mathbf{z}_o$ be the logits of a single external teacher
* $\mathbf{z}_k$ be the logits of external teacher $k$
* $T$ be the distillation temperature

---

## External Teacher Target

For a single teacher:

$$
p_o^T =
\operatorname{softmax}\left(\frac{\mathbf{z}_o}{T}\right)
$$

For an ensemble of $K$ teachers:

$$
p_o^T
=====

\frac{1}{K}
\sum_{k=1}^{K}
\operatorname{softmax}
\left(
\frac{\mathbf{z}_k}{T}
\right)
$$

Combined:

$$
p_o^T=
\begin{cases}
\operatorname{softmax}(\mathbf{z}*o/T),
& \text{single teacher},[2mm]
\displaystyle
\frac{1}{K}
\sum*{k=1}^{K}
\operatorname{softmax}(\mathbf{z}_k/T),
& \text{teacher ensemble}.
\end{cases}
$$

The ensemble target is obtained by averaging the **temperature-scaled probability distributions**, rather than simply averaging hard predictions.

---

# Teacher and Student Distributions

The active teacher distribution is

$$
p_t^T=
\begin{cases}
\operatorname{softmax}(\mathbf{z}_i/T),
& \text{aligned-teacher stage},\
p_o^T,
& \text{external-teacher stage}.
\end{cases}
$$

The student distribution is

$$
p_s^T
=====

\operatorname{softmax}
\left(
\frac{\mathbf{z}_s}{T}
\right).
$$

This allows the same distillation objective to switch between the aligned teacher and the stronger external teacher ensemble during different training phases.

---

# Intermediate Feature Matching

Feature supervision is applied at a subset of structurally aligned layers.

Let

$$
\mathcal{S}
$$

be the set of selected aligned layers.

We use

$$
|\mathcal{S}| = 4.
$$

This corresponds to roughly one feature-loss position for every two FusionConv blocks.

Using more feature-loss locations did not provide additional gains and increased training time.

---

# Feature Normalization

Before computing the feature loss, both student and teacher representations are $\ell_2$-normalized.

For student features:

$$
\widehat{\mathbf{F}}_{s}^{(l)}
==============================

\frac{
\mathbf{F}*{s}^{(l)}
}{
\left|
\mathbf{F}*{s}^{(l)}
\right|_2
+
\varepsilon
}
$$

For teacher features:

$$
\widehat{\mathbf{F}}_{i}^{(l)}
==============================

\frac{
\mathbf{F}*{i}^{(l)}
}{
\left|
\mathbf{F}*{i}^{(l)}
\right|_2
+
\varepsilon
}
$$

Normalization reduces differences in absolute activation scale and makes feature supervision focus more strongly on representation structure.

---

# Huber Feature Loss

The normalized intermediate representations are matched using element-wise Huber loss:

$$
\mathcal{L}_{\mathrm{feat}}
===========================

\frac{1}{|\mathcal{S}|}
\sum_{l\in\mathcal{S}}
\frac{1}{N_l}
\sum_{j=1}^{N_l}
\rho_\delta
\left(
\widehat{f}_{s,j}^{(l)}
-----------------------

\operatorname{sg}
\left(
\widehat{f}_{i,j}^{(l)}
\right)
\right)
$$

where:

* $N_l$ is the number of elements in feature map $l$
* $\operatorname{sg}(\cdot)$ stops gradients through the teacher
* $\varepsilon$ provides numerical stability
* $\delta=1$

The Huber function is

$$
\rho_\delta(u)
==============

\begin{cases}
\frac{1}{2}u^2,
& |u|\leq\delta,[2mm]
\delta
\left(
|u|-\frac{\delta}{2}
\right),
& \text{otherwise}.
\end{cases}
$$

Huber loss behaves quadratically for small differences and linearly for large differences, making feature supervision less sensitive to large representation mismatches than pure MSE.

---

# Complete Training Objective

The complete student objective is

$$
\mathcal{L}
===========

(1-\alpha)
\mathcal{L}*{\mathrm{CE}}
(\mathbf{z}*s,y)
+
\alpha T^2
\mathcal{L}*{\mathrm{KL}}
\left(
p_t^T
\parallel
p_s^T
\right)
+
\beta
\mathcal{L}*{\mathrm{feat}}.
$$

The three components are:

### 1. Ground-truth supervision

$$
(1-\alpha)
\mathcal{L}_{\mathrm{CE}}
(\mathbf{z}_s,y)
$$

Standard cross-entropy keeps the student tied to the true class labels.

### 2. Logit-level teacher supervision

$$
\alpha T^2
\mathcal{L}_{\mathrm{KL}}
\left(
p_t^T
\parallel
p_s^T
\right)
$$

KL divergence transfers the softer class relationships learned by the active teacher.

### 3. Representation-level supervision

$$
\beta
\mathcal{L}_{\mathrm{feat}}
$$

The student is also trained to reproduce normalized intermediate representations from the structurally aligned teacher.

The coefficients control the balance:

* $\alpha$: label supervision vs. teacher-logit supervision
* $\beta$: intermediate-feature supervision
* $T$: temperature of the softened class distribution

---

# Three-Phase Training Strategy

## Phase I — Train the Aligned Teacher

The aligned teacher is first trained independently.

| Dataset      | Epochs | Accuracy |
| ------------ | -----: | -------: |
| TAU          |     60 |    58.3% |
| ESC-50       |     51 |    86.6% |
| UrbanSound8K |     63 |    90.3% |

---

## Phase II — Representation + Logit Transfer

The compact student is trained using:

* ground-truth labels
* aligned-teacher logits
* normalized intermediate features
* Huber feature matching

Parameters:

$$
\alpha=0.6
$$

$$
\beta=0.8
$$

$$
T=5
$$

These values are kept fixed rather than being tuned independently for every dataset.

Phase II results:

| Dataset      | Epochs | Student Accuracy |
| ------------ | -----: | ---------------: |
| TAU          |     66 |            57.8% |
| ESC-50       |     68 |            86.5% |
| UrbanSound8K |     72 |            89.5% |

At this stage, the small student already approaches the performance of the larger aligned teacher.

---

## Phase III — Strong External Guidance

After the student has learned the aligned teacher's representations, training shifts toward stronger output-level supervision.

Parameters:

$$
T=8
$$

$$
\alpha=0.95
$$

$$
\beta=0
$$

Feature matching is disabled:

$$
\mathcal{L}_{\mathrm{feat}} = 0.
$$

The final optimization is therefore dominated by teacher logits while retaining a small amount of direct label supervision.

| Dataset      | Phase III Epochs | Accuracy Before FP16 Conversion | FP16 Drop |
| ------------ | ---------------: | ------------------------------: | --------: |
| TAU          |               51 |                           60.3% |  ~0.10 pp |
| ESC-50       |               65 |                           93.5% |  ~0.08 pp |
| UrbanSound8K |               52 |                           94.5% |  ~0.12 pp |

Final deployed results are approximately:

* **TAU: 60.24%**
* **ESC-50: 93.4%**
* **UrbanSound8K: 94.43%**

---

# Why Staging Helps

The training procedure separates two different objectives.

During the first student stage, the model learns **how the aligned teacher represents the input**.

During the final stage, it learns **how stronger external models distribute probability across classes**.

In simplified form:

```text
Representation Transfer
       ↓
Student becomes structurally stronger
       ↓
Prediction Transfer
       ↓
Student approaches stronger teachers
       ↓
Teachers removed
       ↓
Tiny standalone model
```

The inference architecture never becomes larger as a result of distillation.

---

# External Teacher Ensemble

For the final guidance stage, five models are selected from a larger candidate pool and their temperature-scaled predictions are averaged.

The selected teacher configurations are:

* LEN-v1
* LEN-v2
* CP-Mobile
* CPResNet
* HTS-AT

For ESC-50 and UrbanSound8K, teacher models are initialized using large-scale audio pretraining and then fine-tuned separately on the target folds.

The ensemble consistently provides the strongest student result.

---

# Input Processing

All audio is converted to:

```text
Mono
32 kHz
```

Mel-spectrogram extraction uses:

| Setting       |  Value |
| ------------- | -----: |
| Sampling rate | 32 kHz |
| FFT size      |   4096 |
| Hop length    |    502 |
| Mel bins      |    256 |
| Power         |    2.0 |
| Dynamic range |  80 dB |

Resulting input dimensions:

| Dataset      | Mel Spectrogram Shape |
| ------------ | --------------------: |
| TAU          |              256 × 64 |
| UrbanSound8K |             256 × 255 |
| ESC-50       |             256 × 319 |

For UrbanSound8K:

* clips shorter than 4 seconds are zero-padded
* clips longer than 4 seconds are trimmed

---

# Data Augmentation

The training pipeline combines offline and online augmentation.

## Offline

Device and room-response augmentation is performed using microphone impulse responses.

## During Training

The following augmentations are applied:

* time masking
* frequency masking
* random gain
* additive Gaussian noise
* MixStyle
* device-response augmentation

This is particularly useful for acoustic-scene classification where recording-device mismatch can heavily affect performance.

---

# Optimization

All runs use:

| Setting                             |            Value |
| ----------------------------------- | ---------------: |
| Optimizer                           |             Adam |
| Initial learning rate               |        $10^{-3}$ |
| Batch size                          |               64 |
| Scheduler                           | Cosine annealing |
| Validation split for early stopping |              20% |
| Random seed                         |            Fixed |
| Training GPU                        |  RTX 3090, 24 GB |

Training stops when validation accuracy no longer improves.

---

# Complexity and Main Ablation

The complete training pipeline produces substantial gains over the same compact architecture trained directly from labels.

| Dataset      | Teacher MMACs | Teacher Params | Teacher Size | Student MMACs | Student Params | Student Size | Scratch | External KD Only | No Feature Loss | No Feature Norm. | **Full** |
| ------------ | ------------: | -------------: | -----------: | ------------: | -------------: | -----------: | ------: | ---------------: | --------------: | ---------------: | -------: |
| TAU          |         144.6 |         140.1K |     273.7 KB |      **28.6** |      **31.2K** |  **62.5 KB** |    54.1 |             59.1 |            59.9 |             60.0 | **60.2** |
| ESC-50       |         722.9 |         145.3K |     283.8 KB |     **143.2** |      **36.4K** |  **72.8 KB** |    79.2 |             89.8 |            91.5 |             92.7 | **93.4** |
| UrbanSound8K |         578.3 |         140.1K |     273.7 KB |     **114.5** |      **31.2K** |  **62.5 KB** |    80.0 |             92.2 |            93.1 |             94.3 | **94.4** |

The gains relative to training from scratch are substantial:

| Dataset      | Scratch |      Full | Absolute Gain |
| ------------ | ------: | --------: | ------------: |
| TAU          |   54.1% | **60.2%** |   **+6.1 pp** |
| ESC-50       |   79.2% | **93.4%** |  **+14.2 pp** |
| UrbanSound8K |   80.0% | **94.4%** |  **+14.4 pp** |

These improvements come **without changing the student's inference architecture**.

---

# Feature Loss Ablation

Multiple feature-matching objectives were tested.

Phase II accuracy:

| Dataset      |   L1 | **Huber** | Cosine | Normalized MSE |   KL |  MSE |
| ------------ | ---: | --------: | -----: | -------------: | ---: | ---: |
| TAU          | 56.4 |  **57.8** |   56.2 |           55.3 | 54.7 | 56.8 |
| ESC-50       | 85.2 |  **86.5** |   84.4 |           83.3 | 84.7 | 84.8 |
| UrbanSound8K | 87.8 |  **89.5** |   85.5 |           87.9 | 84.6 | 86.9 |

Huber performs best on all three datasets and is therefore used for intermediate feature matching.

---

# Teacher Selection Ablation

Each cell below shows:

```text
Teacher Accuracy
Student Accuracy
```

The reported decrease is the student accuracy loss when the aligned-teacher stage is removed and only conventional output-level distillation is used.

## TAU

| Teacher      | Teacher Acc. | Student Acc. | Drop Without Aligned Teacher |
| ------------ | -----------: | -----------: | ---------------------------: |
| LEN-v1       |         57.1 |         56.4 |                       3.4 pp |
| LEN-v2       |         55.8 |         56.2 |                       2.6 pp |
| CP-Mobile    |         54.3 |         55.1 |                       3.7 pp |
| CPResNet     |         51.9 |         53.4 |                       3.2 pp |
| HTS-AT       |         52.1 |         53.3 |                       4.1 pp |
| **Ensemble** |     **63.6** |     **60.2** |                   **1.1 pp** |

## ESC-50

| Teacher      | Teacher Acc. | Student Acc. | Drop Without Aligned Teacher |
| ------------ | -----------: | -----------: | ---------------------------: |
| LEN-v1       |         89.2 |         88.2 |                       4.0 pp |
| LEN-v2       |         86.0 |         86.8 |                       4.3 pp |
| CP-Mobile    |         88.9 |         88.3 |                       4.1 pp |
| CPResNet     |         87.6 |         87.7 |                       4.2 pp |
| HTS-AT       |         93.5 |         90.1 |                       3.9 pp |
| **Ensemble** |     **96.6** |     **93.4** |                   **3.6 pp** |

## UrbanSound8K

| Teacher      | Teacher Acc. | Student Acc. | Drop Without Aligned Teacher |
| ------------ | -----------: | -----------: | ---------------------------: |
| LEN-v1       |         90.1 |         90.0 |                       3.5 pp |
| LEN-v2       |         86.3 |         88.2 |                       3.7 pp |
| CP-Mobile    |         91.2 |         90.9 |                       3.3 pp |
| CPResNet     |         90.0 |         89.1 |                       3.5 pp |
| HTS-AT       |         95.8 |         93.6 |                       2.6 pp |
| **Ensemble** |     **97.3** |     **94.4** |                   **2.2 pp** |

The ensemble gives the strongest final result on all three datasets.

---

# TAU Low-Complexity Comparison

On the low-complexity TAU deployment setup, the compact model achieves the highest accuracy among the compared systems while also having the lowest parameter count and smallest memory footprint.

| System         |  Accuracy | Parameters |     MMACs |         Size | Precision |
| -------------- | --------: | ---------: | --------: | -----------: | --------: |
| Baseline       |     51.5% |      61.1K |     29.42 |    122.30 KB |    16-bit |
| Ens-Gui-St     |     59.9% |      60.0K |     30.00 |     121.0 KB |    16-bit |
| TFSN           |     58.1% |     126.9K |     29.42 |    507.43 KB |    32-bit |
| Linear-C       |     59.1% |      63.9K |     29.84 |    127.75 KB |    16-bit |
| NEPUMSE        |     58.3% |     107.5K | **16.91** |    429.83 KB |    32-bit |
| **This model** | **60.2%** |  **31.2K** |     28.60 | **62.52 KB** |    16-bit |

Compared with the smallest competing model in this table:

```text
60.0K parameters → 31.2K parameters
121.0 KB          → 62.5 KB
59.9% accuracy    → 60.2% accuracy
```

In other words, the model is **smaller while also being more accurate**.

---

# DCASE 2025 Official Evaluation

The model was also evaluated in the per-device setting.

Results:

| Metric          |  Accuracy |
| --------------- | --------: |
| **Overall**     | **62.3%** |
| Known devices   | **64.3%** |
| Unknown devices | **58.4%** |

The system exceeds the first-ranked comparison system by approximately:

$$
+0.7 \text{ percentage points}
$$

while using roughly half as many parameters.

![DCASE 2025 comparison](images/Dcase2025.png)

Source path:

```text
images/Dcase2025.png
```

---

# ESC-50 and UrbanSound8K Comparison

Despite its extremely small parameter count, the compact configuration remains competitive with systems containing millions or hundreds of millions of parameters.

| Model             | Max Parameters |          ESC-50 |     UrbanSound8K |
| ----------------- | -------------: | --------------: | ---------------: |
| BEATs iter3       |           300M |            95.6 |             86.1 |
| Dasheng 0.6B      |           600M |            88.2 |             85.8 |
| MATPAC++          |            86M |            93.1 |             89.7 |
| M2D-CLAP          |           149M |        **97.9** |             89.7 |
| ITFA-DNN          |             2M |            94.2 |             95.3 |
| SpectroMaskNet    |           2.7M |           95.50 |        **96.32** |
| AudioPG           |            86M |           90.60 |            88.17 |
| PP-KD (EAT-S)     |          5.18M |           83.80 |            81.90 |
| S-SONDO (DyMN)    |          8.70M |           91.90 |            86.20 |
| SSATKD (HLTDNN)   |          12.3K |           82.65 |                — |
| Micro CNN-PSK     |          50.8K |           86.50 |            84.52 |
| MQaKD (VGG-11)    |         208.6K |           80.03 |            94.95 |
| **Compact model** |       **~36K** |  **93.4 ± 0.9** |   **94.4 ± 1.9** |
| **350K model**    |       **350K** | **95.8 ± 0.83** | **96.25 ± 1.72** |

The important point is not only the absolute accuracy.

The compact configuration reaches **93.4% on ESC-50 and 94.4% on UrbanSound8K with only tens of thousands of parameters**, while many competitive systems operate in the millions or hundreds of millions.

The larger **350K** variant reaches:

* **95.8%** on ESC-50
* **96.25%** on UrbanSound8K

while remaining far smaller than many large audio-classification systems.

---

# Parameter Scaling

Increasing student depth also increases the size of the structurally aligned teacher.

Accuracy generally improves as capacity is increased, but the gains begin to saturate around:

* approximately **230K parameters** on ESC-50
* approximately **300K parameters** on UrbanSound8K

The scaling experiment compares:

* student trained from scratch
* full staged training
* student with external-teacher distillation
* student with aligned-teacher supervision

Figure source:

[Open parameter-scaling source](my_plot2.tex)

```text
my_plot2.tex
```

---

# Compact vs. Larger Variant

Two operating points are especially useful.

## Ultra-Compact

```text
~31K–36K parameters
```

Best when deployment size and memory are the main constraints.

Results:

```text
TAU             60.24%
ESC-50          93.4%
UrbanSound8K    94.43%
```

## Higher-Capacity

```text
350K parameters
```

Best when additional capacity is available while still requiring a relatively small network.

Results:

```text
ESC-50          95.8 ± 0.83%
UrbanSound8K    96.25 ± 1.72%
```

---

# FP16 Deployment

Converting the final student to FP16 approximately halves model storage while causing only a very small accuracy decrease.

Observed drops are approximately:

| Dataset      | FP16 Accuracy Drop |
| ------------ | -----------------: |
| TAU          |            0.10 pp |
| ESC-50       |            0.08 pp |
| UrbanSound8K |            0.12 pp |

For the 10-class compact model:

```text
Parameters : 31.2K
FP16 Size  : ~62.5 KB
```

This makes the architecture suitable for:

* embedded audio systems
* mobile inference
* wearable devices
* always-on acoustic classification
* memory-constrained edge hardware
* low-power audio applications

---

# Why the Method Works

The performance gains come from combining several ideas rather than relying on a single form of distillation.

## 1. Structural Compatibility

The teacher and student share compatible feature dimensions.

This avoids the mismatch that normally makes intermediate feature distillation difficult.

---

## 2. Stronger Teacher Capacity Where It Matters

FusionConv increases the capacity of the teacher while preserving structural compatibility with the student.

---

## 3. Representation Transfer Before Strong Output Transfer

The student first learns intermediate representations from the aligned teacher before receiving stronger external-teacher supervision.

---

## 4. Normalized Feature Matching

Feature normalization removes much of the raw activation-scale mismatch between networks.

---

## 5. Robust Feature Loss

Huber loss provides more stable feature matching than the alternative losses evaluated here.

---

## 6. Soft Probability Transfer

Temperature-scaled distributions expose relationships between classes that are hidden by hard labels.

---

## 7. Teacher Ensembles

Multiple teacher outputs can be averaged to provide a stronger and more stable target distribution.

---

## 8. Zero Teacher Cost at Deployment

Every teacher is discarded after training.

The deployed network remains exactly the compact student:

```text
Training
Student + Aligned Teacher + External Teachers
                    │
                    ▼
                Distillation
                    │
                    ▼
Deployment
              Student Only
```

---

# Summary

This framework demonstrates that strong audio-classification performance does not require a massive inference model.

A compact student with only **31.2K parameters** can reach:

* **60.24%** on TAU
* **93.4%** on ESC-50
* **94.43%** on UrbanSound8K

while occupying approximately **62.5 KB in FP16** for the 10-class configuration.

The main strategy is to make training powerful while keeping inference extremely small:

```text
Structurally Aligned Teacher
            +
Normalized Huber Feature Matching
            +
Temperature-Scaled Logit Distillation
            +
External Teacher Ensemble
            +
Staged Optimization
            ↓
     Tiny High-Accuracy Student
```

The result is a practical audio-classification system capable of reaching **state-of-the-art low-complexity performance on TAU** and highly competitive performance on ESC-50 and UrbanSound8K with dramatically fewer parameters than many larger alternatives.
