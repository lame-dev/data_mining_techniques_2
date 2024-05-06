import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, ndcg_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/training_set_VU_DM.csv')

X = df.drop(['booking_bool', 'click_bool'], axis=1)  # drop non-feature columns
y = df['booking_bool']

# check if there are any categorical features
categorical_features = X.select_dtypes(include=['object']).columns
print("Categorical features: ", categorical_features)
#drop date time columns
X = X.drop(['date_time'], axis=1)

# Encode categorical features if any, here's an example of how to encode:
# label_encoder = LabelEncoder()
# X['some_categorical_feature'] = label_encoder.fit_transform(X['some_categorical_feature'])

# Initialize XGBoost classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict_proba(X_test)[:, 1]

relevance_estimates = y_pred / max(y_pred)  # Normalizing to a scale of 0 to 1
relevance_estimates = relevance_estimates * 5  # Scaling up to 0 to 5

true_relevance = y_test.to_numpy()
true_relevance_sorted = [x for _, x in sorted(zip(y_pred, true_relevance), reverse=True)]

ndcg = ndcg_score([true_relevance_sorted], [relevance_estimates[:5]], k=5)
print("NDCG@5 Score: ", ndcg)