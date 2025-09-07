# Iris Classifier — Clean, Reproducible Notebook

**Short:** A tidy, well-documented Iris classifier notebook (Jupyter `.ipynb`) that demonstrates correct ML workflow: EDA → train/test split → scaling → model training → hyperparameter tuning → evaluation — with emphasis on avoiding overfitting and showing reproducible results.

---

## Project overview

This repo contains a single-end project intended for learning and portfolio use. It builds notebook into a proper, reproducible pipeline using `scikit-learn` and documents findings clearly.

**Goals**

* Demonstrate correct ML workflow and reproducibility.
* Train a robust classifier on the Iris dataset.
* Evaluate with proper metrics.
* Explain limitations and next steps (no fake 100% claims).

---

## Dataset

* File included: `dataSets/IRIS.csv` (150 rows, 4 features + target).
* Source: Classic Iris dataset (UCI / scikit-learn).
* Target classes: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`.

---

## Repository structure 

```
.
├── README.md
├── dataSets/
│   └── IRIS.csv
├── notebook/
│   └── iris_classifier.ipynb
└── requirements.txt
 
```

---

## Dependencies

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## How to run

1. Activate the virtual environment .
2. Start Jupyter:

```bash
jupyter lab        # or jupyter notebook
```

3. Open `notebook/iris_classifier.ipynb`.
4. Run cells top-to-bottom.

---

## Notebook highlights (what it contains)

* **Data loading & sanity checks** — head, dtypes, class balance, missing values.
* **Visual EDA** — pairplot / feature distributions to show class separability.
* **Train/Test split** — `train_test_split(..., stratify=y, random_state=42, test_size=0.2)`.
* **Scaling** — `StandardScaler` fit on `X_train` then transform test set.
* **Model training**:

  * `RandomForestClassifier` with `GridSearchCV` hyperparameter tuning.
* **Evaluation**:

  * `accuracy_score`, `confusion_matrix`, `classification_report`.
* **Feature importance** (Random Forest) visualized and printed.
* **Short discussion** about overfitting, model choice, and limitations.
* **Reproducibility**: `random_state=42` used where applicable.

---

## Reproduce the exact reported results

The notebook includes the code that produced these results on the current run (example output included for transparency):

```
Accuracy: 0.966667  (96.6667%)

Classification Report:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      0.90      0.95        10
 Iris-virginica       0.91      1.00      0.95        10

       accuracy                           0.97        30
      macro avg       0.97      0.97      0.97        30
   weighted avg       0.97      0.97      0.97        30
```

> NOTE: Because the dataset is small, exact test-split accuracy can vary slightly depending on the random seed or the train/test split. Use the cross-validation cell to get a more stable performance estimate.

---

## Recommended commands & snippets (copy-paste)

Train with GridSearch (already in notebook):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)
```


## Key findings & interpretation

* `Iris-setosa` is linearly separable → always near-perfect precision/recall.
* `Iris-versicolor` and `Iris-virginica` overlap in feature space → most errors happen between these two classes.
* Final tuned Random Forest gave \~96.7% accuracy on the held-out 20% test split — excellent for a toy dataset but **do not** overinterpret small improvements beyond this point: risk of overfitting is high.
* Cross-validation should be used to report robust performance (mean ± std).

---

## Limitations

* Small dataset (150 samples) → high variance in single-split metrics.
* Classic toy problem: not representative of real-world messy data.
* Model interpretability: Random Forest gives feature importances but is still an ensemble (less interpretable than a single tree/logistic model).

---

## Next steps (if you want to beef it up)

* Run `StratifiedKFold` cross-validation and report averaged metrics.
* Try `SVC` with RBF kernel and `XGBoost` / `LightGBM` (requires adding their packages) for comparison.
* Add a small holdout or bootstrap strategy for extra robustness.
* Create a `requirements.txt` lockfile with pinned versions (e.g., `pip freeze > requirements.txt`).
* Add a short `results/` folder with PNGs: confusion matrix, feature importances, and pairplot.

---


## Contact

* LinkedIn: www.linkedin.com/in/murali-krishna-m893
* Email: murali.krishna1591@gmail.com

---
