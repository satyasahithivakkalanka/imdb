import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from collections import Counter
import joblib
import json

# trying to import boosted tree models and handling absence gracefully
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    has_histgb = True
except Exception:
    HistGradientBoostingRegressor = None
    has_histgb = False

try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    xgb = None
    has_xgb = False


# defining helper for calculating rmse while supporting older sklearn versions
def rmse_metric(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))


# loading dataset
print("Loading dataset")
df = pd.read_csv("movie_metadata.csv")

# dropping duplicates and rows with missing target values
df = df.drop_duplicates().dropna(subset=["imdb_score"])

# coercing numeric columns and creating a clean numeric base
numeric_candidates = [
    "num_critic_for_reviews", "duration", "gross", "num_voted_users",
    "cast_total_facebook_likes", "facenumber_in_poster", "num_user_for_reviews",
    "budget", "title_year", "movie_facebook_likes", "director_facebook_likes",
    "actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes",
    "aspect_ratio"
]
for c in numeric_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# removing impossible monetary values when both budget and gross exist
if "budget" in df.columns and "gross" in df.columns:
    df.loc[df["budget"] < 0, "budget"] = np.nan
    df.loc[df["gross"] < 0, "gross"] = np.nan

# defining safe log transform
def safe_log1p(s):
    return np.log1p(s.clip(lower=0))

# creating log features
print("Creating log based features")
df["log_budget"] = safe_log1p(df["budget"]) if "budget" in df.columns else 0.0
df["log_gross"] = safe_log1p(df["gross"]) if "gross" in df.columns else 0.0
df["log_votes"] = safe_log1p(df["num_voted_users"]) if "num_voted_users" in df.columns else 0.0
df["log_user_reviews"] = safe_log1p(df["num_user_for_reviews"]) if "num_user_for_reviews" in df.columns else 0.0
df["log_critic_reviews"] = safe_log1p(df["num_critic_for_reviews"]) if "num_critic_for_reviews" in df.columns else 0.0
df["log_movie_fb"] = safe_log1p(df["movie_facebook_likes"]) if "movie_facebook_likes" in df.columns else 0.0
df["log_cast_fb"] = safe_log1p(df["cast_total_facebook_likes"]) if "cast_total_facebook_likes" in df.columns else 0.0

# creating ratio and interaction features
print("Creating ratio and interaction features")
if "gross" in df.columns and "budget" in df.columns:
    df["roi_ratio"] = np.where(df["budget"] > 0, df["gross"] / df["budget"], np.nan)
    df["log_roi_ratio"] = safe_log1p(df["roi_ratio"].replace([np.inf, -np.inf], np.nan))
else:
    df["roi_ratio"] = np.nan
    df["log_roi_ratio"] = 0.0

if "duration" in df.columns and "num_voted_users" in df.columns:
    df["votes_per_minute"] = np.where(df["duration"] > 0, df["num_voted_users"] / df["duration"], np.nan)
    df["log_votes_per_minute"] = safe_log1p(df["votes_per_minute"].replace([np.inf, -np.inf], np.nan))
else:
    df["log_votes_per_minute"] = 0.0

# creating decade and language or country flags
print("Creating temporal and flag features")
df["decade"] = (df["title_year"] // 10) * 10 if "title_year" in df.columns else np.nan
df["is_english"] = (df["language"].fillna("").str.lower() == "english") if "language" in df.columns else False
df["is_usa"] = (df["country"].fillna("").str.upper() == "USA") if "country" in df.columns else False

# expanding genres into multi hot representation
print("Expanding genres")
genre_features = []
if "genres" in df.columns:
    genre_split = df["genres"].fillna("").str.lower().str.split("|")
    top_genres = [g for g, _ in Counter([x for l in genre_split for x in l if x]).most_common(25)]
    for g in top_genres:
        col = f"genre_{g}"
        df[col] = genre_split.apply(lambda xs: int(g in xs))
        genre_features.append(col)

# expanding plot keywords into multi hot representation
print("Expanding plot keywords")
keyword_features = []
if "plot_keywords" in df.columns:
    kw_split = df["plot_keywords"].fillna("").str.lower().str.replace("-", " ", regex=False).str.split("|")
    top_kws = [k for k, _ in Counter([x for l in kw_split for x in l if x]).most_common(60)]
    for k in top_kws:
        col = f"kw_{k.replace(' ', '_')}"
        df[col] = kw_split.apply(lambda xs: int(k in xs))
        keyword_features.append(col)

# selecting final feature lists
numeric_features = [
    "duration", "aspect_ratio", "facenumber_in_poster",
    "log_budget", "log_gross", "log_votes", "log_user_reviews",
    "log_critic_reviews", "log_movie_fb", "log_cast_fb",
    "log_roi_ratio", "log_votes_per_minute"
]
numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = []
if "color" in df.columns: categorical_features.append("color")
if "content_rating" in df.columns: categorical_features.append("content_rating")
if "language" in df.columns: categorical_features.append("language")
if "country" in df.columns: categorical_features.append("country")
if "decade" in df.columns: categorical_features.append("decade")

bool_features = ["is_english", "is_usa"]

all_features = numeric_features + categorical_features + bool_features + genre_features + keyword_features

# creating feature and target matrices
X = df[all_features]
y = df["imdb_score"].astype(float)

# creating preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

# passing boolean and already multi hot features with numeric branch
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features + bool_features + genre_features + keyword_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} rows and testing on {len(X_test)} rows")

# defining base models
models = {
    "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5),
    "RandomForest": RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
}

if has_histgb:
    models["HistGB"] = HistGradientBoostingRegressor(
        max_depth=None, learning_rate=0.06, max_iter=700,
        l2_regularization=0.0, random_state=42
    )

if has_xgb:
    models["XGB"] = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

# defining search spaces for boosted models
search_spaces = {}
if has_histgb:
    from scipy.stats import loguniform, randint
    search_spaces["HistGB"] = {
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_depth": randint(3, 12),
        "model__max_iter": randint(300, 1200),
        "model__l2_regularization": loguniform(1e-8, 1e-1),
        "model__max_leaf_nodes": randint(15, 63)
    }

if has_xgb:
    from scipy.stats import loguniform, randint
    search_spaces["XGB"] = {
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_depth": randint(3, 10),
        "model__n_estimators": randint(400, 1400),
        "model__subsample": loguniform(0.5, 1.0),
        "model__colsample_bytree": loguniform(0.5, 1.0),
        "model__reg_lambda": loguniform(1e-3, 10.0)
    }

# defining evaluation helper
def evaluate_pipeline(name, pipeline, X_tr, y_tr, X_te, y_te):
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    rmse = rmse_metric(y_te, preds)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1).mean()
    return r2, mae, rmse, cv_r2

# training base models and collecting metrics
results = {}
best_model = None
best_r2 = -np.inf

for name in ["RidgeCV", "RandomForest"]:
    print(f"Training {name}")
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", models[name])])
    r2, mae, rmse, cv_r2 = evaluate_pipeline(name, pipe, X_train, y_train, X_test, y_test)
    results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "cv_r2": cv_r2}
    print(f"{name} R2 {r2:.4f} MAE {mae:.3f} RMSE {rmse:.3f} CV_R2 {cv_r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipe

# running randomized searches for boosted models when available
for name in ["HistGB", "XGB"]:
    if name in models and name in search_spaces:
        print(f"Running randomized search for {name}")
        base = Pipeline(steps=[("prep", preprocessor), ("model", models[name])])
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=search_spaces[name],
            n_iter=30,
            scoring="r2",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        preds = search.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = rmse_metric(y_test, preds)
        results[f"{name}_Tuned"] = {
            "r2": r2, "mae": mae, "rmse": rmse, "cv_r2": float(max(search.best_score_, -1))
        }
        print(f"{name}_Tuned R2 {r2:.4f} MAE {mae:.3f} RMSE {rmse:.3f} CV_R2 {results[f'{name}_Tuned']['cv_r2']:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = search.best_estimator_

# saving artifacts and predictions
Path("artifacts").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, "artifacts/imdb_best_model.joblib")
with open("artifacts/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

best_preds = best_model.predict(X_test)
pd.DataFrame({
    "imdb_score_actual": y_test.values,
    "imdb_score_pred": best_preds
}).to_csv("artifacts/test_predictions.csv", index=False)

print("Training complete")
print(f"Best R2 on test {best_r2:.4f}")
print("Saving metrics model and predictions into artifacts folder")
