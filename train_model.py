import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'housing.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")

    # 1. Imputasi nilai kosong total_bedrooms dengan median (sesuai Bab 3 praktikum)
    median_bedrooms = float(df['total_bedrooms'].median())
    df_clean = df.copy()
    df_clean['total_bedrooms'] = df_clean['total_bedrooms'].fillna(median_bedrooms)

    features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
    ]
    target = 'median_house_value'

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    cat_cols = ['ocean_proximity']
    num_cols = [c for c in features if c != 'ocean_proximity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    # Random Forest Regressor
    model_rf = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training Random Forest Regressor...")
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    print(f"Model Regresi: R2 = {r2:.4f}, MAE = ${mae:,.2f}, RMSE = ${rmse:,.2f}, MAPE = {mape:.2f}%")

    # Feature Importance
    encoder = model_rf.named_steps['prep'].named_transformers_['cat']
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    all_feature_names = cat_feature_names + num_cols
    importances = model_rf.named_steps['reg'].feature_importances_

    feat_imp = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # 2. Model K-Means Clustering (Sesuai Bab 6 Buku Abu Tholib)
    print("Training K-Means Clustering (Bab 6)...")
    coords = df_clean[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    kmeans.fit(coords)
    df_clean['cluster'] = kmeans.labels_

    # Evaluasi silhouette score pada sampel 3.000 titik (biar cepat)
    sample_idx = np.random.RandomState(42).choice(len(coords), size=3000, replace=False)
    sil_score = float(silhouette_score(coords.iloc[sample_idx], kmeans.labels_[sample_idx]))
    print(f"K-Means Silhouette Score: {sil_score:.4f}")

    # Klaster stats
    cluster_stats = {}
    for cl in range(3):
        subset = df_clean[df_clean['cluster'] == cl]
        mean_val = float(subset['median_house_value'].mean())
        mean_lat = float(subset['latitude'].mean())
        mean_lon = float(subset['longitude'].mean())
        # Tentukan label geografis
        if mean_lat > 37.0:
            region = "California Utara (Bay Area / Sacramento)"
        elif mean_lat > 34.5:
            region = "California Tengah (Central Valley / Fresno)"
        else:
            region = "California Selatan (Greater Los Angeles / San Diego)"

        cluster_stats[str(cl)] = {
            "region": region,
            "mean_price": mean_val,
            "total_blocks": len(subset),
            "mean_latitude": mean_lat,
            "mean_longitude": mean_lon,
            "price_level": "Tinggi" if mean_val > 230000 else ("Menengah" if mean_val > 170000 else "Lebih Rendah")
        }

    # Simpan model pickle
    rf_path = os.path.join(base_dir, 'model_rf.pkl')
    km_path = os.path.join(base_dir, 'model_kmeans.pkl')

    with open(rf_path, 'wb') as f:
        pickle.dump(model_rf, f)
    with open(km_path, 'wb') as f:
        pickle.dump(kmeans, f)

    # Simpan metrics json
    metrics = {
        "r2_score": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "median_bedrooms_imputed": median_bedrooms,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_importances": feat_imp,
        "kmeans_silhouette": sil_score,
        "cluster_stats": cluster_stats,
        "categories": sorted(list(df['ocean_proximity'].unique())),
        "price_summary": {
            "min": float(df['median_house_value'].min()),
            "max": float(df['median_house_value'].max()),
            "mean": float(df['median_house_value'].mean()),
            "median": float(df['median_house_value'].median())
        }
    }

    metrics_path = os.path.join(base_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print("Successfully saved model_rf.pkl, model_kmeans.pkl, and metrics.json!")

if __name__ == '__main__':
    main()
