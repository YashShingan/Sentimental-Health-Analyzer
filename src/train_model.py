import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor # Usually better for this scale than RF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. Load & Clean
df = pd.read_csv("../data/social_media_dataset.csv")

# 2. FEATURE ENGINEERING (The Secret Sauce)
# Addiction isn't just usage; it's usage relative to sleep.
df['usage_sleep_ratio'] = df['avg_daily_usage_hours'] / (df['sleep_hours_per_night'] + 1)

# 3. Select Features including Categorical
features = ["age", "avg_daily_usage_hours", "sleep_hours_per_night", "usage_sleep_ratio", "most_used_platform"]
X = df[features]
y = df["addicted_score"]

# 4. Handle Categorical Data (One-Hot Encoding)
X = pd.get_dummies(X, columns=["most_used_platform"])

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model - Gradient Boosting often outperforms RF for tabular health data
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
accuracy = r2_score(y_test, model.predict(X_test))
print(f"Improved Model R2 Score: {accuracy:.4f}")

# 8. Save Model and Column structure (Crucial for Streamlit)
joblib.dump(model, "../models/addiction_model.pkl")
joblib.dump(X.columns.tolist(), "../models/model_columns.pkl")