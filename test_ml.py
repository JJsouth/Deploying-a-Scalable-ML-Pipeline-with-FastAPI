import pytest
import pandas as pd
import numpy as np
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
    save_model,
    load_model,
)
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

CAT_FEATURES = ["workclass", "education"]


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 38, 28, 44, 18, 34, 29, 63],
            "hours-per-week": [40, 50, 40, 45, 20, 40, 60, 35],
            "workclass": [
                "Private",
                "Private",
                "Local-gov",
                "Private",
                "?",
                "Self-emp",
                "Private",
                "Local-gov",
            ],
            "education": [
                "HS-grad",
                "Masters",
                "Bachelors",
                "Masters",
                "HS-grad",
                "Bachelors",
                "Masters",
                "HS-grad",
            ],
            "salary": [
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                "<=50K",
                "<=50K",
                ">50K",
                "<=50K",
            ],
        }
    )


@pytest.fixture
def processed(sample_df):
    X, y, _, _ = process_data(
        sample_df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y


@pytest.fixture
def model(processed):
    X, y = processed
    return train_model(X, y)


def test_compute_model_metrics_known_values():
    """Precision, recall, and F1 match a hand-computed example."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1)
    assert recall == pytest.approx(1 / 2)
    assert fbeta == pytest.approx(2 / 3)


def test_train_model_returns_random_forest(model):
    """train_model returns a fitted RandomForestClassifier."""
    assert isinstance(model, RandomForestClassifier)


def test_inference_output(model, processed):
    """inference returns one 0/1 prediction per input row."""
    X, _ = processed
    preds = inference(model, X)

    assert preds.shape == (len(X),)
    assert set(preds) <= {0, 1}


def test_save_load_roundtrip(model, processed, tmp_path):
    """Trained model matches loaded model."""
    path = tmp_path / "model.pkl"
    X, _ = processed
    save_model(model, str(path))
    loaded_model = load_model(str(path))

    assert (model.predict(X) == loaded_model.predict(X)).all()
