import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import pickle

MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'subject_encoder.pkl'

# a session is "productive" if focus >= 4
PRODUCTIVITY_THRESHOLD = 4


def build_features(df):
    """
    Prepare the feature matrix X and label vector y from a DataFrame.
    Features: hour of day, subject (encoded), mood, duration.
    """
    le = LabelEncoder()
    df = df.copy()
    df['subject_encoded'] = le.fit_transform(df['subject'])

    X = df[['hour', 'subject_encoded', 'mood', 'duration']].values
    y = (df['focus'] >= PRODUCTIVITY_THRESHOLD).astype(int).values

    return X, y, le


def train(sessions):
    """
    Train a Random Forest on the session data.
    Returns a results dict with accuracy and class balance info.
    """
    from analytics import sessions_to_df

    df = sessions_to_df(sessions)

    if df.empty or len(df) < 5:
        return {'error': 'Need at least 5 sessions to train the model.'}

    X, y, le = build_features(df)

    # if all sessions have the same label the model can't learn anything useful
    if len(np.unique(y)) < 2:
        return {'error': 'Need a mix of productive and non-productive sessions to train.'}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)

    # save the trained model and encoder so we don't retrain on every request
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    productive_pct = round(y.mean() * 100, 1)

    return {
        'accuracy': accuracy,
        'total_sessions': len(df),
        'productive_pct': productive_pct,
        'trained': True,
    }


def load_model():
    """Load saved model and encoder from disk. Returns (None, None) if not trained yet."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    return clf, le


def predict(subject, start_time, mood, duration):
    """
    Predict whether a planned session will be productive.
    Returns a dict with the prediction and confidence %.
    """
    clf, le = load_model()

    if clf is None:
        return {'error': 'Model not trained yet. Go to the dashboard first.'}

    hour = int(start_time.split(':')[0])

    # handle subjects not seen during training
    if subject not in le.classes_:
        # use the mean encoded value as a rough fallback
        subject_encoded = int(np.mean(le.transform(le.classes_)))
    else:
        subject_encoded = le.transform([subject])[0]

    features = np.array([[hour, subject_encoded, mood, duration]])
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]

    confidence = round(max(probabilities) * 100, 1)
    label = 'Productive' if prediction == 1 else 'Low Productivity'

    # map confidence into a plain-English description
    if confidence >= 80:
        confidence_label = 'High confidence'
    elif confidence >= 60:
        confidence_label = 'Moderate confidence'
    else:
        confidence_label = 'Low confidence'

    return {
        'prediction': label,
        'productive': bool(prediction),
        'confidence': confidence,
        'confidence_label': confidence_label,
        'productive_prob': round(probabilities[1] * 100, 1) if len(probabilities) > 1 else None,
    }


def get_feature_importance(sessions):
    """
    Return feature importances as a list so we can display them on the dashboard.
    Useful for explaining what the model actually learned.
    """
    from analytics import sessions_to_df

    clf, le = load_model()
    if clf is None:
        return []

    feature_names = ['Hour of Day', 'Subject', 'Mood', 'Duration']
    importances = clf.feature_importances_

    result = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return [{'feature': name, 'importance': round(imp * 100, 1)} for name, imp in result]
