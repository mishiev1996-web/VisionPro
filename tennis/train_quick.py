"""Tennis model training — incremental (no leakage)."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tennis.tennis_db as tennis_db
from tennis.features import TennisFeatureEngine, FEATURE_NAMES

# Build features INCREMENTALLY (no load_history = no leakage)
print("Building features incrementally...")
engine = TennisFeatureEngine()

rows = []
with tennis_db.connect() as conn:
    matches = conn.execute("""
        SELECT player1_id, player2_id, winner_id, surface, date,
               player1_name, player2_name,
               w_1stIn, w_svpt, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
               l_1stIn, l_svpt, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced,
               (SELECT ranking FROM tennis_players WHERE id=player1_id),
               (SELECT ranking FROM tennis_players WHERE id=player2_id)
        FROM tennis_matches 
        WHERE status='finished' AND date IS NOT NULL
        ORDER BY date ASC
    """).fetchall()

for m in matches:
    p1_id, p2_id, winner_id = m[0], m[1], m[2]
    surface, date = m[3] or 'hard', m[4]
    p1_rank, p2_rank = m[19], m[20]

    # Serve stats: w_1stIn(7), w_svpt(8), w_1stWon(9), w_2ndWon(10), w_bpSaved(11), w_bpFaced(12)
    #              l_1stIn(13), l_svpt(14), l_1stWon(15), l_2ndWon(16), l_bpSaved(17), l_bpFaced(18)

    # Get features BEFORE updating
    feats = engine.get_features(p1_id, p2_id, surface, p1_rank, p2_rank, date)
    label = 1 if winner_id == p1_id else 0
    rows.append({**feats, 'label': label, 'date': date})

    # Determine winner/loser
    if winner_id == p1_id:
        w_id, l_id = p1_id, p2_id
        w_prefix, l_prefix = 7, 13  # indices for winner/loser serve stats
    else:
        w_id, l_id = p2_id, p1_id
        w_prefix, l_prefix = 13, 7  # swapped

    # Build serve stats for winner and loser
    def _to_f(v):
        try:
            return float(v) if v is not None else 0.0
        except:
            return 0.0

    def _srv_pct(svpt, won):
        s, w = _to_f(svpt), _to_f(won)
        return w / s * 100 if s > 0 else 50.0

    def _bp_pct(saved, faced):
        s, f = _to_f(saved), _to_f(faced)
        return s / f * 100 if f > 0 else 50.0

    w_stats = {
        "1stWon": _srv_pct(m[w_prefix + 1], m[w_prefix + 2]),
        "2ndWon": _srv_pct(_to_f(m[w_prefix + 1]) - _to_f(m[w_prefix + 0]), m[w_prefix + 3]),
        "bpSaved": _bp_pct(m[w_prefix + 4], m[w_prefix + 5]),
    }
    l_stats = {
        "1stWon": _srv_pct(m[l_prefix + 1], m[l_prefix + 2]),
        "2ndWon": _srv_pct(_to_f(m[l_prefix + 1]) - _to_f(m[l_prefix + 0]), m[l_prefix + 3]),
        "bpSaved": _bp_pct(m[l_prefix + 4], m[l_prefix + 5]),
    }

    # Update engine WITH serve stats
    engine.update(w_id, l_id, surface, date, serve_stats={w_id: w_stats, l_id: l_stats})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=FEATURE_NAMES + ['date'])

# Random flip
np.random.seed(42)
flip_mask = np.random.random(len(df)) < 0.5
for col in FEATURE_NAMES:
    df.loc[flip_mask, col] = -df.loc[flip_mask, col]
df.loc[flip_mask, 'label'] = 1 - df.loc[flip_mask, 'label']

print(f"Dataset: {len(df)} rows")
print(f"Win rate: {df['label'].mean():.3f}")

# Split
split_date = df['date'].quantile(0.8)
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
print(f"Train: {len(train)}, Test: {len(test)}")

X_train = train[FEATURE_NAMES].values
y_train = train['label'].values
X_test = test[FEATURE_NAMES].values
y_test = test['label'].values

# Train
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.85, colsample_bytree=0.85, random_state=42,
                     eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)

lgbm = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                       subsample=0.85, colsample_bytree=0.85, random_state=42,
                       verbose=-1)
lgbm.fit(X_train, y_train)

# Evaluate
xgb_prob = xgb.predict_proba(X_test)[:, 1]
lgbm_prob = lgbm.predict_proba(X_test)[:, 1]
ens_prob = (xgb_prob + lgbm_prob) / 2

xgb_pred = (xgb_prob > 0.5).astype(int)
lgbm_pred = (lgbm_prob > 0.5).astype(int)
ens_pred = (ens_prob > 0.5).astype(int)

print(f"\n=== Results (no leakage) ===")
print(f"XGBoost:   acc={accuracy_score(y_test, xgb_pred):.4f}  ll={log_loss(y_test, xgb_prob):.4f}")
print(f"LightGBM:  acc={accuracy_score(y_test, lgbm_pred):.4f}  ll={log_loss(y_test, lgbm_prob):.4f}")
print(f"Ensemble:  acc={accuracy_score(y_test, ens_pred):.4f}  ll={log_loss(y_test, ens_prob):.4f}")

print(f"\nTop features:")
imp = sorted(zip(FEATURE_NAMES, xgb.feature_importances_), key=lambda x: -x[1])
for name, score in imp:
    print(f"  {name:25s} {score:.4f}")
