# Wine Quality Classification

This repository contains the code and report for the classification of wine quality using machine learning models. The primary objective of this project was to compare the computational and predictive performance of **Random Forest (RF)** and **Support Vector Machine (SVM)** classifiers on the **Wine Quality Dataset**. The task involved addressing the challenges posed by an imbalanced dataset.

---

## Dataset
The **White Wine Quality Dataset** was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The dataset consists of 11 physicochemical features (e.g., acidity, alcohol content, pH) and a target variable (wine quality) with 7 classes (3 to 9). The dataset exhibits significant class imbalance, with most samples rated as "6" and "5," while classes like "3," "9," and "4" are underrepresented.

### Class Distribution
| Quality | Percentage |
|---------|------------|
| 6       | 44.88%     |
| 5       | 29.75%     |
| 7       | 17.97%     |
| 8       | 3.57%      |
| 4       | 3.33%      |
| 3       | 0.41%      |
| 9       | 0.10%      |

---

## Methodology

### Preprocessing Steps
1. **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets.
2. **Scaling**: Features were standardized using `StandardScaler` to ensure uniform scaling.
3. **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to address the class imbalance and improve predictions for minority classes.

### Classifiers Used
1. **Random Forest (RF)**:
   - Chosen for its robustness, ability to handle non-linear relationships, and resistance to overfitting.
2. **Support Vector Machine (SVM)**:
   - Selected for its effectiveness in high-dimensional spaces and capability to classify imbalanced datasets with appropriate kernel and parameter tuning.

### Cross-Validation Strategy
- **Repeated k-Fold Cross-Validation** with `n_splits=3` and `n_repeats=10` was employed to ensure reliable and unbiased performance estimates by reducing variability due to random splits.

### Evaluation Metrics
The following metrics were used to evaluate model performance:
- **Accuracy**
- **Precision (Macro)**
- **Recall (Macro)**
- **F1-Score (Macro)**

---

## Results

### Cross-Validation Results (Imbalanced Dataset)
| Metric            | SVM               | Random Forest     |
|-------------------|-------------------|-------------------|
| **Accuracy**      | 61.36% (\u00b11.03%) | 64.40% (\u00b11.00%) |
| **Precision**     | 47.26% (\u00b13.84%) | 51.61% (\u00b13.88%) |
| **Recall**        | 32.66% (\u00b12.45%) | 34.64% (\u00b12.65%) |
| **F1-Score**      | 35.72% (\u00b12.64%) | 38.04% (\u00b13.02%) |

- **Random Forest** outperformed SVM across all metrics, particularly in **F1-Score** and **Recall**.

### Cross-Validation Results (Balanced Dataset with SMOTE)
| Metric            | Random Forest (Balanced) |
|-------------------|---------------------------|
| **Accuracy**      | 89.06% (\u00b10.50%)      |
| **Precision**     | 88.66% (\u00b10.51%)      |
| **Recall**        | 89.06% (\u00b10.48%)      |
| **F1-Score**      | 88.74% (\u00b10.50%)      |

- After balancing the dataset, performance improved significantly across all metrics.

### Final Model Results on Test Set
| Metric            | Imbalanced Test Set | Balanced Test Set |
|-------------------|---------------------|-------------------|
| **Accuracy**      | 69.08%              | 66.63%            |
| **Precision**     | 52.14%              | 40.08%            |
| **Recall**        | 36.84%              | 40.84%            |
| **F1-Score**      | 40.17%              | 40.32%            |

- Balancing the dataset slightly decreased test accuracy and precision but improved recall and F1-Score, highlighting better generalization for minority classes.

---

## Discussion
- Balancing the dataset using SMOTE significantly improved cross-validation metrics, especially for minority classes.
- The final model showed a trade-off between prioritizing overall accuracy and improving performance for underrepresented classes.
- **Random Forest** consistently outperformed **SVM**, making it the better choice for this task.

---

## Conclusion
This project demonstrated the impact of dataset balancing on machine learning performance. While **Random Forest** proved to be the most effective classifier overall, balancing the dataset using SMOTE enhanced the model's ability to generalize across all classes, particularly minority classes. These findings emphasize the importance of addressing class imbalance in machine learning tasks.

---

## References
1. Dua, D. and Graff, C. "UCI Machine Learning Repository: Wine Quality dataset," UCI Machine Learning Repository, 2019.
2. Google Developers, "Imbalanced datasets," Google Machine Learning Crash Course.
