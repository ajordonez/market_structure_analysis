"""Features and models for predicting next-day BTC vs ETH outperformance."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import SEED

N_LAGS = 7                   # days of lagged returns used as features
C_VALUES = np.logspace(-3, 2, 11)


def make_lag_features(returns, n_lags=N_LAGS):
    # Row t: returns on days t-1 .. t-n_lags. Target: 1 if BTC's return beats ETH's on day t.
    features = pd.DataFrame({f"{asset.lower()}_lag{k}": returns[asset].shift(k)
                             for asset in ["BTC", "ETH"] for k in range(1, n_lags + 1)})
    target = (returns["BTC"] > returns["ETH"]).astype(int).rename("btc_outperforms")
    data = pd.concat([features, target], axis=1).dropna()
    return data.drop(columns=target.name), data[target.name]


def error_rate(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def make_logreg(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C,
                           max_iter=5000, random_state=SEED),
    )


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=2, random_state=SEED, n_jobs=-1)


def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                         random_state=SEED, verbosity=0, n_jobs=1)


class Net(nn.Module):
    # One-hidden-layer MLP that outputs a logit.
    def __init__(self, n_features, hidden=11):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x)


def train_net(X_tr, y_tr, X_va, y_va, max_epochs=500, patience=20, lr=1e-3, seed=SEED):
    # Full-batch Adam with early stopping on validation loss. Returns (predict_fn, best_epoch).
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    as_tensor = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    Xt, Xv = as_tensor(scaler.transform(X_tr)), as_tensor(scaler.transform(X_va))
    yt, yv = as_tensor(y_tr).unsqueeze(1), as_tensor(y_va).unsqueeze(1)

    model = Net(Xt.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(Xt), yt).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return (model(as_tensor(scaler.transform(X))) > 0).int().numpy().ravel()

    return predict, best_epoch


def fit_predict_all(X_tr, y_tr, X_te):
    # Fit every model on X_tr (tuning on its last 20%) and return predictions for X_te.
    cut = int(len(X_tr) * 0.8)
    X_fit, y_fit, X_in, y_in = X_tr.iloc[:cut], y_tr.iloc[:cut], X_tr.iloc[cut:], y_tr.iloc[cut:]

    inner_errors = [error_rate(y_in, make_logreg(C).fit(X_fit, y_fit).predict(X_in)) for C in C_VALUES]
    C = C_VALUES[int(np.argmin(inner_errors))]
    nn_fn, _ = train_net(X_fit, y_fit, X_in, y_in)

    return {
        "Naive Baseline": np.full(len(X_te), int(y_tr.mean() >= 0.5)),
        "Logistic Regression": make_logreg(C).fit(X_tr, y_tr).predict(X_te),
        "Random Forest": make_rf().fit(X_tr, y_tr).predict(X_te),
        "XGBoost": make_xgb().fit(X_tr, y_tr).predict(X_te),
        "Neural Network": nn_fn(X_te),
    }
