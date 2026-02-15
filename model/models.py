from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb

def load_data(data):
    data['quality_label'] = (data['quality'] >= 6).astype(int)
    X = data.drop(['quality', 'quality_label'], axis=1)
    y = data['quality_label']

    # Train/test split
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

def evaluate_model(name, model, X_test, y_test, y_pred):
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return metrics

def get_confusion_metrics(name, y_test, y_pred):
    rows = []
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
