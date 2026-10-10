"""Models: a lightweight NumPy autoencoder and scikit-learn risk scorers."""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import config as cfg


def stack_ensemble(scores_train, y_train, seed=cfg.RANDOM_SEED):
    """Learn a logistic-regression meta-learner over the three detector scores.

    ``scores_train`` is an (n, 3) array of [autoencoder error, isolation
    score, random-forest probability].  The stacked model generalises soft
    majority voting and typically matches or exceeds its best base learner.
    Scores are standardised first because they live on very different scales.
    """
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    meta.fit(scores_train, y_train)
    return meta


# --------------------------------------------------------------------------- #
# Autoencoder (self-contained NumPy implementation for full reproducibility)
# --------------------------------------------------------------------------- #
class Autoencoder:
    """Undercomplete autoencoder (input -> hidden -> latent -> hidden -> output).

    Trained on the *on-track* (Good + Average) class so reconstruction error
    can serve as a one-class score for the Poor (at-risk) group.
    """

    def __init__(self, input_dim, hidden=cfg.AE_HIDDEN_DIM,
                 latent=cfg.AE_LATENT_DIM, lr=cfg.AE_LEARNING_RATE,
                 seed=cfg.RANDOM_SEED):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = self._init(input_dim, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = self._init(hidden, latent)
        self.b2 = np.zeros(latent)
        self.W3 = self._init(latent, hidden)
        self.b3 = np.zeros(hidden)
        self.W4 = self._init(hidden, input_dim)
        self.b4 = np.zeros(input_dim)

    def _init(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, (fan_in, fan_out))

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.01 * z1)                 # leaky ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)                               # latent
        z3 = a2 @ self.W3 + self.b3
        a3 = np.maximum(z3, 0.01 * z3)
        z4 = a3 @ self.W4 + self.b4
        return z4, (z1, a1, z2, a2, z3, a3, z4)

    def _backward(self, X, cache):
        z1, a1, z2, a2, z3, a3, z4 = cache
        n = X.shape[0]
        d4 = (z4 - X) / n
        gW4 = a3.T @ d4
        gb4 = d4.sum(0)
        d3 = (d4 @ self.W4.T) * np.where(z3 > 0, 1, 0.01)
        gW3 = a2.T @ d3
        gb3 = d3.sum(0)
        d2 = (d3 @ self.W3.T) * (1 - a2 ** 2)
        gW2 = a1.T @ d2
        gb2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * np.where(z1 > 0, 1, 0.01)
        gW1 = X.T @ d1
        gb1 = d1.sum(0)
        return [(gW1, gb1), (gW2, gb2), (gW3, gb3), (gW4, gb4)]

    def fit(self, X, epochs=cfg.AE_EPOCHS, verbose=False):
        X = np.asarray(X, dtype=float)
        # Flatten the parameter list (W1,b1,W2,b2,W3,b3,W4,b4)
        params = [self.W1, self.b1, self.W2, self.b2,
                  self.W3, self.b3, self.W4, self.b4]
        # Adam optimizer state
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        for e in range(epochs):
            out, cache = self._forward(X)
            grads_pairs = self._backward(X, cache)
            flat_grads = [g for pair in grads_pairs for g in pair]
            t += 1
            for i, (g, p) in enumerate(zip(flat_grads, params)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
                mh = m[i] / (1 - beta1 ** t)
                vh = v[i] / (1 - beta2 ** t)
                p -= self.lr * mh / (np.sqrt(vh) + eps)
            if verbose and e % 10 == 0:
                mse = np.mean((out - X) ** 2)
                print(f"  epoch {e:3d}  mse={mse:.5f}")
        return self

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                self.W4, self.b4]

    def set_weights(self, weights):
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3, self.W4, self.b4) = weights

    def reconstruct(self, X):
        out, _ = self._forward(np.asarray(X, dtype=float))
        return out

    def anomaly_scores(self, X):
        """Per-sample mean squared reconstruction error."""
        out = self.reconstruct(X)
        return np.mean((np.asarray(X, dtype=float) - out) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# Ensemble (soft majority voting of the three detectors)
# --------------------------------------------------------------------------- #
class FederatedEnsemble:
    """Combines autoencoder, Isolation Forest and Random Forest predictions.

    A sample is flagged as an anomaly if at least two of the three detectors
    agree (soft majority voting).
    """

    def __init__(self, ae, iforest, rf):
        self.ae = ae
        self.iforest = iforest
        self.rf = rf

    def predict(self, X):
        ae_score = self.ae.anomaly_scores(X)
        ae_pred = (ae_score > np.percentile(ae_score, 60)).astype(int)
        if_pred = (self.iforest.predict(X) == -1).astype(int)
        rf_pred = self.rf.predict(X)
        votes = ae_pred + if_pred + rf_pred
        return (votes >= 2).astype(int)


# --------------------------------------------------------------------------- #
# Isolation Forest / Random Forest factories
# --------------------------------------------------------------------------- #
def build_isolation_forest(n_estimators=cfg.IF_ESTIMATORS,
                           contamination=cfg.ANOMALY_CONTAMINATION,
                           seed=cfg.RANDOM_SEED):
    return IsolationForest(n_estimators=n_estimators,
                           contamination=contamination,
                           random_state=seed, n_jobs=1)


def build_random_forest(n_estimators=cfg.RF_ESTIMATORS, seed=cfg.RANDOM_SEED):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  class_weight="balanced",
                                  random_state=seed, n_jobs=1)
