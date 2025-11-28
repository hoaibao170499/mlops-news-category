import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("news_classification")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Bước 1: Chuyển kiểu viết thường
    result = text.lower()
    # Bước 2: Xoá noisy words (number, punctuation, special chars, html,...) bằng Regular Expression.
    result = re.sub(r'\d+', '', result)
    result = re.sub(r'<\/?\w+>', ' ', result)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[^\w\s]', '', result)
    # Bước 3: Tách từ (tokenization)
    tokens = nltk.word_tokenize(result)
    # Bước 4: Xoá các từ dừng (stopwords)
    tokens = [token for token in tokens if token not in stop_words]
    # Bước 5: Stemming & Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_data(row):
    return row.astype(str).apply(preprocess_text)

def train():
    mlflow.set_experiment("news_classification_experiment")
    # Paths
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "20newsgroups.csv"
    ARTIFACT_DIR = PROJECT_ROOT / "scripts" 
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "20newsgroups_cls_model.joblib"

    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    logger.info("Preparing features and target...")
    # Identify target and basic features from the CSV header
    TARGET = "label"
    NUM_FEATURES = "posts"

    X = df[NUM_FEATURES]
    y = df[TARGET]

    logger.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = FunctionTransformer(preprocess_data)

    logger.info("Building pipeline...")
    model = Pipeline(
        steps = [
            ("preprocessing", preprocessor),
            ("vectorize", TfidfVectorizer()),
            ("classify", MultinomialNB()),
        ]
    )

    logger.info("Training model...")
    with mlflow.start_run(run_name="news_classification_1"):
        model.fit(X_train, y_train)

        # Evaluate model performance
        logger.info("Evaluating model performance...")

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1_score = f1_score(y_train, y_train_pred, average='weighted')

        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1_score = f1_score(y_test, y_test_pred, average='weighted')

        # Log training metrics
        logger.info("Training set metrics:")
        logger.info(f"  Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Precision: {train_precision:.4f}")
        logger.info(f"  Recall: {train_recall:.4f}")
        logger.info(f"  F1_Score: {train_f1_score:.4f}")

        # Log test metrics
        logger.info("Test set metrics:")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1_Score: {test_f1_score:.4f}")

        # Log model performance summary
        logger.info("Model performance summary:")
        logger.info(
            f"  Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )
        logger.info(
            f"  Training F1_Score: {train_f1_score:.4f}, Test F1_Score: {test_f1_score:.4f}"
        )

        # save the model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to: {MODEL_PATH}")

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1_score", train_f1_score)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1_score)
        mlflow.log_artifact(MODEL_PATH, "artifacts")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()