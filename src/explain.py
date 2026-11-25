import numpy as np
import shap


class Explainer:
    def __init__(self, xgb_model, feature_names, classes):
        self.exp = shap.TreeExplainer(xgb_model)
        self.features = list(feature_names)
        self.classes = list(classes)

    def _values(self, X) -> np.ndarray:
        """-> array (n_samples, n_features, n_classes) for any shap version."""
        sv = self.exp.shap_values(X)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        if sv.ndim == 2:
            sv = sv[..., None]
        return sv

    def top_features(self, X, class_idx, k=3):
        """Per-row top-k features pushing toward class_idx[row]."""
        sv = self._values(X)
        out = []
        for i in range(len(X)):
            v = sv[i, :, class_idx[i]]
            order = np.argsort(-np.abs(v))[:k]
            out.append([{"feature": self.features[j],
                         "value": round(float(X.iloc[i, j]), 3),
                         "shap": round(float(v[j]), 3)} for j in order])
        return out

    def global_importance(self, X):
        """Mean |SHAP| per feature across samples and classes (global view)."""
        sv = np.abs(self._values(X)).mean(axis=(0, 2))
        return sorted(zip(self.features, sv), key=lambda t: -t[1])
