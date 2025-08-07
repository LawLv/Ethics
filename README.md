# Computer Ethics Project: Depth to Fairness

## 📌 Project Overview

This project focuses on **toxic comment classification** using deep learning models, while assessing and mitigating **algorithmic bias** across various demographic subgroups. The work is divided into two main parts:

1. **Benchmark Evaluation** – Training a baseline CNN model and assessing subgroup fairness.
2. **Bias Mitigation** – Improving fairness using preprocessing and attention-based LSTM architecture.

---


## 🔧 Benchmark Model

- **Model**: Convolutional Neural Network (CNN)
- **Task**: Classify toxic comments
- **Evaluation**: Accuracy & fairness across subgroups (AUC, BPSN AUC, BNSP AUC)

### Key Findings

| Subgroup | Subgroup AUC | BPSN AUC | BNSP AUC |
|----------|--------------|----------|----------|
| homosexual_gay_or_lesbian | 0.7962 | 0.7738 | 0.9533 |
| black                     | 0.7998 | 0.7677 | 0.9565 |
| muslim                    | 0.8199 | 0.8169 | 0.9458 |
| white                     | 0.8216 | 0.7690 | 0.9616 |
| female                    | 0.8803 | 0.8784 | 0.9362 |
| christian                 | 0.8935 | 0.9135 | 0.9165 |

- Final benchmark fairness-aware metric: **0.8843**
- **Issues Identified**: Lower AUCs in subgroups like *black* and *homosexual_gay_or_lesbian*, indicating potential bias.

---

## 💡 Value Alignment

Ten example comments were manually evaluated and compared to model predictions. Results showed high agreement between human judgments and the model, confirming baseline model consistency.

---

## ⚖️ Demographic Parity Assessment (Benchmark)

- **Overall Positive Rate**: 6.9%
- **Subgroup disparities** (largest shown):
  - `black`: +16.37%
  - `muslim`: +9.72%
  - `christian`: +0.14% (minimal disparity)

---

## 🛠️ Bias Mitigation & Model Improvement

### 1. Preprocessing with Fairlearn

- **Method**: `CorrelationRemover` from Fairlearn
- **Targeted Bias**: Strong correlations between `black`/`muslim` and features like `identity_attack`
- **Outcome**: Eliminated these correlations while preserving other relationships

### 2. Model Architecture

- **Model**: Two-layer LSTM with Self-Attention and Residual Connections
- **Implementation**: See `lstm_attention_7.ipynb`
- **Training Time**: ~2800s per epoch

### 3. Results

| Metric        | Benchmark | Improved Model |
|---------------|-----------|----------------|
| Final Metric  | 0.8843    | **0.9268**     |
| Disparity (black) | 0.1637 | **0.1238**     |
| Disparity (white) | 0.1563 | **0.1146**     |

Subgroup AUCs improved across the board, especially for previously underperforming groups.

---

## 📊 Demographic Parity (After Mitigation)

| Subgroup | Disparity Difference ↓ | Disparity Ratio ↑ |
|----------|------------------------|-------------------|
| christian | 0.0194                | 0.7083            |
| black     | 0.1238                | 0.2757            |
| muslim    | 0.0882                | 0.4154            |

---

## 🧠 Conclusions

- Attention-based LSTM with preprocessing significantly improves **fairness** and **overall performance**.
- Certain subgroups still show disparities (e.g., *black*), suggesting room for further work.
- Fairlearn tools proved effective for bias mitigation in real datasets.

---

## 📚 Dependencies

- Python 3.8+
- TensorFlow / Keras
- Fairlearn
- Scikit-learn
- Pandas / Numpy / Matplotlib

Install via:

```bash
pip install -r requirements.txt
