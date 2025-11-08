# IMDB Movie Rating Prediction

This project predicts IMDB movie ratings (⁠ imdb_score ⁠) using advanced feature engineering and multiple regression models.  
It processes metadata like budget, gross, duration, Facebook likes, genres, and keywords to train machine learning models that generalize well across movies.

---

##  Project Overview

This pipeline automates:

### *Data Cleaning & Preprocessing*
•⁠  ⁠Removes duplicates and rows with missing targets.
•⁠  ⁠Handles invalid or negative numeric values.

### *Feature Engineering*
•⁠  ⁠Log transformations for numeric skewed data.
•⁠  ⁠Derived ratios and interaction terms (e.g., ROI).
•⁠  ⁠Temporal features like decade.
•⁠  ⁠Binary flags for language and country.

### *Feature Expansion*
•⁠  ⁠One-hot encodes top 25 genres and top 60 plot keywords.

### *Model Training & Evaluation*
•⁠  ⁠Ridge Regression, Random Forest, HistGradientBoosting, and XGBoost.
•⁠  ⁠Performs Randomized Search CV for boosted models.
•⁠  ⁠Selects the best model based on R² score.

### *Artifact Saving*
•⁠  ⁠Exports trained model, metrics, and test predictions in ⁠ /artifacts ⁠.

---

##  Models Used

| Model | Description | Tuned | CV Strategy | Library |
|--------|--------------|--------|--------------|-----------|
| RidgeCV | Linear model with cross-validated regularization |  Auto | 5-Fold | scikit-learn |
| RandomForest | Ensemble of decision trees |  Default | 5-Fold | scikit-learn |
| HistGB | Gradient Boosting using histograms |  RandomizedSearchCV | 3-Fold | scikit-learn |
| XGB | Extreme Gradient Boosting |  RandomizedSearchCV | 3-Fold | xgboost |

---

##  Metrics Interpretation (Run Results)

| Model | R² (Test) | MAE | RMSE | CV_R² |
|--------|------------|------|--------|--------|
| RidgeCV | 0.4497 | 0.610 | 0.828 | 0.4474 |
| RandomForest | 0.5846 | 0.514 | 0.719 | 0.5493 |
| HistGB (Tuned) | 0.6132 | 0.490 | 0.694 | 0.5799 |
| XGB (Tuned) |  *0.6173* | *0.485* | *0.691* | *0.5967* |

---

##  Interpretation

*Best Model:* XGBoost (Tuned)  
•⁠  ⁠Achieved *R² = 0.6173*, meaning it explains ~61.7% of the variance in IMDB scores.  
•⁠  ⁠*MAE = 0.485* indicates an average prediction error of ~0.5 rating points.  
•⁠  ⁠*RMSE = 0.691* confirms stable predictions with low variance.  

*Feature Engineering Impact:*  
Log transformations, ratios (like ROI), and genre/keyword expansions helped non-linear models (RF, XGB) outperform linear RidgeCV.

*Performance Warning:*  
The console warns that the DataFrame is highly fragmented due to repeated column insertions.  
 This is only a performance warning, not an error. It can be fixed later by using ⁠ pd.concat() ⁠ for faster joins.

---

##  Requirements
Install the dependencies:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

Optional (for hyperparameter tuning):

```bash
pip install scipy
```

 ⁠
 
---

##  Running the Script

Run in terminal:

```bash
python imdb.py
```
 ⁠

*Output:*


```
Loading dataset
Creating log based features
Creating ratio and interaction features
...
Training RidgeCV
Training RandomForest
Running randomized search for HistGB
Running randomized search for XGB
Training complete
Best R2 on test 0.6173
Saving metrics model and predictions into artifacts folder
```

---

##  Output Artifacts

•⁠  ⁠*metrics.json* → All model scores (R², MAE, RMSE, CV_R²)  
•⁠  ⁠*imdb_best_model.joblib* → Serialized model (can be loaded with ⁠ joblib.load() ⁠)  
•⁠  ⁠*test_predictions.csv* → Comparison between actual and predicted scores
