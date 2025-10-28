
import argparse
import os
import joblib
import pandas as pd
import boto3
import pathlib
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

# SageMaker requires a model_fn to load the model for inference
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == '__main__':
    print("Extracting arguments")
    parser = argparse.ArgumentParser()

    # Define hyperparameters and input data paths as arguments
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')
    parser.add_argument('--train_file', type=str, default='train_V1.csv')
    parser.add_argument('--test_file', type=str, default='test_V1.csv')


    args, _ = parser.parse_known_args()

    # Load data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    # Separate features and target
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    clf.fit(X_train, y_train)

    # Evaluate the model (optional)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_rep = classification_report(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print("Classification report: ")
    print(test_rep)

    # Save the trained model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
