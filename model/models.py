from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def load_data(data):
    X = data['data']
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
    }

    # Train models
    models['Logistic Regression'].fit(X_train_scaled, y_train)
    models['Decision Tree'].fit(X_train, y_train)
    models['K-Nearest Neighbor'].fit(X_train_scaled, y_train)
    models['Naive Bayes'].fit(X_train, y_train)
    models['Random Forest'].fit(X_train, y_train)
    models['XGBoost'].fit(X_train, y_train)

    return scaler, models
