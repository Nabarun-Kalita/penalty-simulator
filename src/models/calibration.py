import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsotonicMultiClassCalibrator:
    def __init__(self):
        self.calibrators = []

    def fit(self, proba, y_true, n_classes):
        self.calibrators = []
        for c in range(n_classes):
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(proba[:, c], (y_true == c).astype(int))
            self.calibrators.append(iso)
        return self

    def predict_proba(self, proba):
        calibrated = np.zeros_like(proba)
        for c, iso in enumerate(self.calibrators):
            calibrated[:, c] = iso.predict(proba[:, c])
        calibrated = np.clip(calibrated, 0.02, 0.98)  # Avoid exact 0 or 1 for stability
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return calibrated / row_sums