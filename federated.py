"""Simulated cross-silo learning components.

The module partitions records into reproducible IID shards. It is a simulation,
not a network implementation: aggregation is performed in-process and no
cryptographic secure-aggregation protocol is implemented.

The autoencoder can add central Gaussian noise to clipped client updates. Its
optional guarantee is client-level (epsilon, delta)-DP for the released
FedAvg autoencoder parameters, using an RDP accountant over all communication
rounds. It does NOT cover RF/IF models, the stacked learner, recommendations,
or information visible to an untrusted aggregation server.
"""

import numpy as np

import config as cfg
from models import Autoencoder, build_random_forest, build_isolation_forest


def make_clients(n, n_clients=cfg.N_CLIENTS, seed=cfg.RANDOM_SEED):
    """Return a reproducible, approximately balanced IID partition of indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, n_clients)


def _clip(update, clip_norm):
    """L2-clip the concatenated client update to ``clip_norm``."""
    norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in update))
    if norm <= clip_norm or norm == 0:
        return [g.copy() for g in update]
    factor = clip_norm / norm
    return [g * factor for g in update]


_RDP_ORDERS = np.array([1.25, 1.5, 1.75, 2, 3, 4, 5, 8, 10, 16,
                        20, 32, 64, 128, 256, 512, 1024], dtype=float)


def rdp_epsilon(noise_multiplier, delta, rounds=cfg.FED_ROUNDS):
    """Convert composed Gaussian RDP to an (epsilon, delta)-DP upper bound.

    ``noise_multiplier`` is standard deviation divided by the L2 sensitivity
    of the aggregated client-level update. No client-subsampling amplification
    is assumed. The finite set of Renyi orders makes this a conservative
    numerical accountant.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    orders = _RDP_ORDERS
    rdp = rounds * orders / (2.0 * noise_multiplier ** 2)
    eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    return float(np.min(eps))


def noise_multiplier_for_epsilon(epsilon, delta=cfg.DP_DELTA,
                                 rounds=cfg.FED_ROUNDS):
    """Find Gaussian noise multiplier meeting the composed RDP bound."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive; use None for no-noise mode")
    lo, hi = 1e-4, 1e4
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if rdp_epsilon(mid, delta, rounds) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def fedavg_autoencoder(clients_X_good, input_dim, rounds=cfg.FED_ROUNDS,
                       epsilon=None, delta=cfg.DP_DELTA,
                       clip_norm=cfg.DP_CLIP_NORM, local_epochs=10,
                       seed=cfg.RANDOM_SEED, return_history=False):
    """Train a FedAvg autoencoder with optional client-level central DP.

    ``epsilon=None`` disables noise and offers no DP guarantee. If epsilon is
    positive, it is the *overall* epsilon for all ``rounds`` under the RDP
    accountant, with ``delta`` also applying to the complete training run.
    Client updates are clipped at ``clip_norm`` and uniformly averaged. Under
    replace-one client adjacency, the released mean has L2 sensitivity
    ``2*clip_norm/n_clients``. Gaussian noise is added to every aggregated
    parameter update before broadcasting the next global model.

    This is central DP at a trusted coordinator; the code does not implement
    cryptographic secure aggregation, so the coordinator can inspect local
    updates. The guarantee covers only the released autoencoder weights.
    """
    rng = np.random.default_rng(seed)
    global_model = Autoencoder(input_dim, seed=seed)
    n_clients = len(clients_X_good)
    if n_clients == 0:
        raise ValueError("at least one client is required")

    sigma = None
    if epsilon is not None:
        sigma = noise_multiplier_for_epsilon(epsilon, delta, rounds)
        sensitivity = 2.0 * clip_norm / n_clients
        noise_std = sigma * sensitivity
    else:
        noise_std = 0.0

    history = []
    nonempty_good = [np.asarray(x, dtype=float) for x in clients_X_good
                     if len(x) > 0]
    eval_good = np.vstack(nonempty_good) if nonempty_good else np.empty((0, input_dim))

    for _ in range(rounds):
        client_updates = []
        global_weights = global_model.get_weights()
        for Xc in clients_X_good:
            if len(Xc) == 0:
                client_updates.append([np.zeros_like(w) for w in global_weights])
                continue
            local = Autoencoder(input_dim, seed=seed)
            local.set_weights([w.copy() for w in global_weights])
            local.fit(Xc, epochs=local_epochs)
            upd = [w - p for w, p in
                   zip(local.get_weights(), global_weights)]
            client_updates.append(_clip(upd, clip_norm))

        # Uniform FedAvg keeps sensitivity independent of data-dependent
        # client weights. Every configured client participates (empty local
        # normal subsets contribute a zero update).
        avg_update = [
            np.mean(np.stack([u[i] for u in client_updates], axis=0), axis=0)
            for i in range(len(global_weights))
        ]
        if noise_std > 0:
            avg_update = [g + rng.normal(0.0, noise_std, size=g.shape)
                          for g in avg_update]
        global_model.set_weights([p + g for p, g in
                                  zip(global_weights, avg_update)])
        if len(eval_good):
            err = global_model.reconstruct(eval_good) - eval_good
            history.append(float(np.mean(err ** 2)))
        else:
            history.append(float("nan"))

    if return_history:
        return global_model, history
    return global_model


def federated_random_forest(clients_X, clients_y, seed=cfg.RANDOM_SEED):
    """Fit client-local RFs; models remain separate (not securely aggregated)."""
    models = []
    for i, (Xc, yc) in enumerate(zip(clients_X, clients_y)):
        if len(Xc) == 0 or len(np.unique(yc)) < 2:
            # A constant-label client cannot fit the RF; callers should ensure
            # sufficiently large shards. The error is preferable to silently
            # substituting a model with a different training objective.
            raise ValueError(f"client {i} needs both binary classes for RF")
        rf = build_random_forest(seed=seed + i)
        rf.fit(Xc, yc)
        models.append(rf)
    return models


def federated_isolation_forest(clients_X, seed=cfg.RANDOM_SEED):
    """Fit client-local Isolation Forests; no parameter aggregation is done."""
    models = []
    for i, Xc in enumerate(clients_X):
        if len(Xc) == 0:
            raise ValueError(f"client {i} is empty")
        iso = build_isolation_forest(seed=seed + i)
        iso.fit(Xc)
        models.append(iso)
    return models


def vote_classify(models, X):
    """Majority vote across a list of fitted classifiers (labels 0/1)."""
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return (preds.mean(axis=0) >= 0.5).astype(int)


def vote_anomaly(models, X):
    """Majority vote across Isolation-Forest detectors (-1 = anomaly)."""
    preds = np.stack([(m.predict(X) == -1).astype(int) for m in models], axis=0)
    return (preds.mean(axis=0) >= 0.5).astype(int)
