import numpy as np

from src.methods import evaluation


def test_smote_mask_uses_imblearn_smote(monkeypatch):
    calls = []

    class DummySMOTE:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit_resample(self, X, y):
            calls.append((X, y))
            return X, y

    monkeypatch.setattr(evaluation, "SMOTE", DummySMOTE)

    features = np.array([[0.0], [0.1], [0.2], [1.0], [1.1]])
    labels = np.array([0, 0, 0, 1, 1])
    expanded_features, expanded_labels, expanded_mask = evaluation.smote_mask(
        np.array([True, True, True, True, True]), features, labels, random_state=42
    )

    assert len(calls) == 1
    assert expanded_features.shape[0] >= features.shape[0]
    assert expanded_labels.shape[0] == expanded_features.shape[0]
    assert expanded_mask.shape[0] == expanded_features.shape[0]
