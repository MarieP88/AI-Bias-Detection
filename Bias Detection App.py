from flask import Flask, request, jsonify
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import BinaryLabelDataset

app = Flask(__name__)


def preprocess_openml_dataset(dataset_name):
    """
    Preprocesses an OpenML dataset, identifying binary and categorical features,
    applying label encoding and one-hot encoding, and returning a transformed DataFrame.

    Args:
        dataset_name (str): The name of the dataset in OpenML.

    Returns:
        tuple: A tuple containing the transformed DataFrame, a list of binary features,
            a list of categorical features, and a list of numerical features.
    """

    # # Fetch the dataset from OpenML
    # dataset = fetch_openml(name=dataset_name, version=1)

    # # Convert to DataFrame
    # df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    # df["target"] = dataset.target
    df = dataset_name
    # Identify binary, categorical, and numerical features
    binary_features = []
    categorical_features = []
    numerical_features = []

    for col in df.columns:
        if col == "target":
            continue
        if df[col].dtype == "object" and df[col].nunique() == 2:
            binary_features.append(col)
        elif df[col].dtype == "object":
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    # Label encode binary features
    label_encoder = LabelEncoder()
    for col in binary_features:
        df[col] = label_encoder.fit_transform(df[col])

    # One-hot encode categorical features
    transformer = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )
    transformed = transformer.fit_transform(df)

    # Create a new DataFrame with transformed features
    transformed_df = pd.DataFrame(
        transformed, columns=transformer.get_feature_names_out()
    )

    return transformed_df, binary_features, categorical_features, numerical_features


# Load datasets
def load_dataset(name):
    if name == "compass":
        # Load Compass dataset
        compas_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        compas_data = pd.read_csv(compas_url)

        # Convert to DataFrame
        dataset = compas_data.drop(columns=["is_recid", "decile_score", "score_text"])
        dataset["target"] = compas_data["is_recid"]

    elif name == "Thyroid_Disease":
        thyroid_disease = fetch_openml(name="thyroid-dis", version=1)
        dataset = pd.DataFrame(
            data=thyroid_disease.data, columns=thyroid_disease.feature_names
        )

        # Convert target to numerical if needed
        if not pd.api.types.is_numeric_dtype(thyroid_disease.target):
            dataset["target"] = pd.Categorical(thyroid_disease.target).codes
        else:
            dataset["target"] = thyroid_disease.target

        dataset["target"] = (dataset.target >= 4).astype(int)

        dataset, cat, bin, num = preprocess_openml_dataset(dataset)
        dataset.rename(columns={"remainder__target": "target"}, inplace=True)
        dataset["target"] = dataset["target"].astype(int)
        # Set the random seed
        np.random.seed(
            42
        )  # Answer to the Ultimate Question of Life, the Universe, and Everything

        dataset["sensitive"] = np.random.choice(
            [0, 1], size=len(dataset), replace=True, p=[0.5, 0.5]
        )

    elif name == "heart_failure":
        heart_failure = fetch_openml(name="heart-failure", version=1)
        dataset = pd.DataFrame(
            data=heart_failure.data, columns=heart_failure.feature_names
        )
        dataset["target"] = dataset.DEATH_EVENT
        dataset = dataset.drop("DEATH_EVENT", axis=1)
        dataset["sensitive"] = dataset.sex

    elif name == "covid":
        # Load German Credit dataset
        covid_data = pd.read_csv("datasets/covid_dataset.csv")
        # Convert to DataFrame
        dataset = covid_data.drop(
            columns=[
                "Update_Date",
                "Record_ID",
                "Admission_Date",
                "Symptoms_Date",
                "Death_Date",
                "Origin_Country",
                "Nationality_Country",
                "Patient_Type",
            ]
        )
        dataset["sensitive"] = (dataset.Sex > 1).astype(int)
        dataset["target"] = (covid_data["Patient_Type"] > 1).astype(int)

    elif name == "german_credit":
        # Load German Credit dataset
        german_credit = pd.read_csv("datasets/german_credit.csv")
        # Convert to DataFrame
        dataset = german_credit.drop(columns=["Creditability"])
        dataset["target"] = german_credit["Creditability"]

    elif name == "breast_cancer":
        # Load Breast Cancer dataset
        breast_cancer = load_breast_cancer()
        # Convert to DataFrame
        dataset = pd.DataFrame(
            data=breast_cancer.data, columns=breast_cancer.feature_names
        )
        # Set the random seed
        np.random.seed(
            42
        )  # Answer to the Ultimate Question of Life, the Universe, and Everything

        dataset["sensitive"] = np.random.choice(
            [0, 1], size=len(dataset), replace=True, p=[0.5, 0.5]
        )

        dataset["target"] = breast_cancer.target

    elif name == "diabetes":
        # Load Diabetes dataset
        diabetes = load_diabetes()
        # Convert to DataFrame
        dataset = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        # Define a threshold (e.g., median of target values)
        threshold = diabetes.target.mean()
        # Convert the target to binary classes
        y_binary = (diabetes.target > threshold).astype(int)
        dataset["target"] = y_binary
        dataset["sensitive"] = (dataset.sex > 0).astype(int)
        # dataset["sensitive_age"] = (dataset.age > 0.05).astype(int)
    else:
        return None

    return dataset


# Calculate Fairness Metrics
def statistical_parity(y_true, y_pred, sensitive_feature):
    if sum(sensitive_feature == 1) == 0:
        return float("inf")
    if sum(sensitive_feature == 0) == 0:
        return float("inf")

    pos_rate_group1 = sum((y_pred == 1) & (sensitive_feature == 1)) / sum(
        sensitive_feature == 1
    )
    pos_rate_group2 = sum((y_pred == 1) & (sensitive_feature == 0)) / sum(
        sensitive_feature == 0
    )
    return pos_rate_group1 - pos_rate_group2


def calculate_equality_of_opportunity(y_true, y_pred, sensitive_feature):
    # Assume sensitive_feature is a binary feature
    sensitive_feature = pd.Series(sensitive_feature).astype(str)

    unique_groups = sensitive_feature.unique()

    tpr_by_group = {}
    for group in unique_groups:
        group_indices = sensitive_feature == group
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_by_group[group] = tpr

    return tpr_by_group


def disparate_impact(y_true, y_pred, sensitive_feature):
    if sum(sensitive_feature == 1) == 0:
        return float("inf")
    if sum(sensitive_feature == 0) == 0:
        return float("inf")

    pos_rate_group1 = sum((y_pred == 1) & (sensitive_feature == 1)) / sum(
        sensitive_feature == 1
    )
    pos_rate_group2 = sum((y_pred == 1) & (sensitive_feature == 0)) / sum(
        sensitive_feature == 0
    )

    result = pos_rate_group1 / pos_rate_group2 if pos_rate_group2 != 0 else float("nan")
    return result


# Plot Orignal Data
def plot_original_dataset(dataset):
    # Features Importance
    X = dataset.drop("target", axis=1)
    y = dataset["target"]

    grouped_df = dataset.groupby(["sensitive", "target"]).size().unstack().fillna(0)

    if y.empty or y.isnull().all():
        raise ValueError("Target variable is empty or contains only NaN values.")

    if len(X) != len(y):
        raise ValueError("Features and target variable have inconsistent lengths.")

    FI_model = ExtraTreesClassifier()
    FI_model.fit(X, y)
    import json

    feat_imp = pd.Series(FI_model.feature_importances_, index=X.columns)
    positives = pd.Series(y, dtype="int64").sum()
    negatives = len(y) - positives

    return {
        "feat_imp": feat_imp.to_dict(),
        "positives": int(positives),
        "negatives": int(negatives),
        "priviliged": sum(dataset.sensitive),
        "unpriviliged": len(dataset) - sum(dataset.sensitive),
        "dataset": dataset.to_dict(orient="records"),
    }


# Preprocess Data
def preprocess_dataset(dataset):
    # Features Importance

    sensitive_feature = dataset["sensitive"]

    transformed_dataset = dataset.drop("target", axis=1)
    transformed_dataset["target"] = dataset["target"]
    transformed_dataset["sensitive"] = sensitive_feature

    # Create BinaryLabelDataset
    transformed_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=transformed_dataset,
        label_names=["target"],
        protected_attribute_names=["sensitive"],
    )

    privileged_groups = [{"sensitive": 1}]
    unprivileged_groups = [{"sensitive": 0}]

    RW = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )

    # Fit and transform the dataset using Reweighing
    RW.fit(transformed_data)
    reweighted_dataset = RW.transform(transformed_data)

    # Extract features, labels, and instance weights
    weights = reweighted_dataset.instance_weights

    priv_weights = weights[sensitive_feature == 1]
    unpriv_weights = weights[sensitive_feature == 0]

    return {
        "weights": weights.tolist(),
        "priv_weights": priv_weights.tolist(),
        "unpriv_weights": unpriv_weights.tolist(),
    }


# Train model
def train_model(dataset, model_type):
    X = dataset.drop("target", axis=1)
    y = dataset["target"]
    sensitive_feature = dataset["sensitive"]

    X_train, X_test, sensitive_train, sensitive_test, y_train, y_test = (
        train_test_split(X, sensitive_feature, y, test_size=0.2, random_state=42)
    )
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC(probability=True)
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "naive_bayes":
        model = GaussianNB()
    else:
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(
        y_test, y_pred
    ).tolist()  # Convert to list for JSON serialization

    # Calculate fairness metrics
    sp = statistical_parity(y_test, y_pred, sensitive_test)
    di = disparate_impact(y_test, y_pred, sensitive_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    tpr_by_group = calculate_equality_of_opportunity(y_test, y_pred, sensitive_test)

    metrics = {
        "cm": cm,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc,
        "disparate": di,
        "statistical": sp,
        "equal_opportunity": tpr_by_group,
    }

    return metrics


# Reweigh & Train model
def reweigh_train_model(dataset, model_type):
    # Split features and target
    X = dataset.drop("target", axis=1)
    y = dataset["target"]
    sensitive_feature = dataset["sensitive"]

    # Split data into training and testing sets
    X_train, X_test, sensitive_train, sensitive_test, y_train, y_test = (
        train_test_split(X, sensitive_feature, y, test_size=0.2, random_state=42)
    )

    # Combine X_train and y_train into a DataFrame for aif360
    train_data = X_train.copy()
    train_data["target"] = y_train
    train_data["sensitive"] = sensitive_train

    # Create BinaryLabelDataset
    train_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=train_data,
        label_names=["target"],
        protected_attribute_names=["sensitive"],
    )

    privileged_groups = [{"sensitive": 1}]
    unprivileged_groups = [{"sensitive": 0}]

    RW = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )

    # Fit and transform the dataset using Reweighing
    RW.fit(train_dataset)
    reweighed_train_dataset = RW.transform(train_dataset)

    # Convert back to DataFrame
    X_train_reweigh = reweighed_train_dataset.features
    y_train_reweigh = reweighed_train_dataset.labels.ravel()

    # Select model
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC(probability=True)
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "naive_bayes":
        model = GaussianNB()
    else:
        return None

    # Train model
    model.fit(X_train_reweigh, y_train_reweigh)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Calculate fairness metrics
    sp = statistical_parity(y_test, y_pred, sensitive_test)
    di = disparate_impact(y_test, y_pred, sensitive_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    tpr_by_group = calculate_equality_of_opportunity(y_test, y_pred, sensitive_test)

    metrics = {
        "cm": cm,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc,
        "disparate": di,
        "statistical": sp,
        "equal_opportunity": tpr_by_group,
    }

    return metrics


@app.route("/")
def index():
    return "Thanks for all the fish"


@app.route("/originaldata", methods=["POST"])
def original_data():
    data = request.get_json()
    dataset_name = data["dataset"]
    dataset = load_dataset(dataset_name)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    preprocess_results = plot_original_dataset(dataset)

    if preprocess_results is None:
        return jsonify({"error": "Dataset not in correct format/ not supported"}), 400

    return jsonify(preprocess_results)


@app.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.get_json()
    dataset_name = data["dataset"]
    dataset = load_dataset(dataset_name)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    preprocess_results = preprocess_dataset(dataset)

    if preprocess_results is None:
        return jsonify({"error": "Dataset not in correct format/ not supported"}), 400

    return jsonify(preprocess_results)


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    dataset_name = data["dataset"]
    model_type = data["model"]
    dataset = load_dataset(dataset_name)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    metrics = train_model(dataset, model_type)
    if metrics is None:
        return jsonify({"error": "Model type not supported"}), 400

    return jsonify(metrics)


@app.route("/reweigh", methods=["POST"])
def reweigh_train():
    data = request.get_json()
    dataset_name = data["dataset"]
    model_type = data["model"]
    dataset = load_dataset(dataset_name)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404
    metrics = reweigh_train_model(dataset, model_type)
    if metrics is None:
        return jsonify({"error": "Model type not supported"}), 400

    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=True)
