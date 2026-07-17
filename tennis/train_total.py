"""Tennis total games — regression + classification approaches."""
import sys, re
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import tennis.tennis_db as tennis_db
from tennis.features import TennisFeatureEngine, TOTAL_FEATURE_NAMES, parse_total_games

print("Building features...")
engine = TennisFeatureEngine()

rows = []
with tennis_db.connect() as conn:
    matches = [tuple(r) for r in conn.execute("""
        SELECT m.player1_id, m.player2_id, m.winner_id, m.surface, m.date, m.score,
               p1.ranking, p2.ranking
        FROM tennis_matches m
        LEFT JOIN tennis_players p1 ON m.player1_id = p1.id
        LEFT JOIN tennis_players p2 ON m.player2_id = p2.id
        WHERE m.status='finished' AND m.date IS NOT NULL AND m.score IS NOT NULL
        ORDER BY m.date ASC
    """).fetchall()]

for m in matches:
    p1_id, p2_id, winner_id = m[0], m[1], m[2]
    surface, date, score = m[3] or 'hard', m[4], m[5]
    p1_rank, p2_rank = m[6], m[7]

    total = parse_total_games(score)
    if total is None:
        continue

    sets = re.findall(r'(\d+)-(\d+)', score)
    n_sets = len(sets) if sets else 2
    is_bo5 = 1 if n_sets > 2 else 0

    feats = engine.get_total_features(p1_id, p2_id, surface, p1_rank, p2_rank, is_bo5, date)
    rows.append({**feats, 'total': total, 'is_bo5': is_bo5, 'date': date})

    if winner_id == p1_id:
        w_id, l_id = p1_id, p2_id
    else:
        w_id, l_id = p2_id, p1_id
    engine.update(w_id, l_id, surface, date, score=score)

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=TOTAL_FEATURE_NAMES + ['date'])

print(f"Dataset: {len(df)} rows")
print(f"Total games: mean={df['total'].mean():.1f}, median={df['total'].median():.0f}, std={df['total'].std():.1f}")

# Split
split_date = df['date'].quantile(0.8)
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
print(f"Train: {len(train)}, Test: {len(test)}")

X_train = train[TOTAL_FEATURE_NAMES].values
y_train_reg = train['total'].values
X_test = test[TOTAL_FEATURE_NAMES].values
y_test_reg = test['total'].values

# ── Approach 1: Regression ──
print(f"\n=== Approach 1: Regression (predict exact total) ===")
xgb_reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.85, colsample_bytree=0.85, random_state=42)
xgb_reg.fit(X_train, y_train_reg)

lgbm_reg = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.85, random_state=42,
                          verbose=-1)
lgbm_reg.fit(X_train, y_train_reg)

xgb_pred_reg = xgb_reg.predict(X_test)
lgbm_pred_reg = lgbm_reg.predict(X_test)
ens_pred_reg = (xgb_pred_reg + lgbm_pred_reg) / 2

print(f"XGBoost:   MAE={mean_absolute_error(y_test_reg, xgb_pred_reg):.2f}")
print(f"LightGBM:  MAE={mean_absolute_error(y_test_reg, lgbm_pred_reg):.2f}")
print(f"Ensemble:  MAE={mean_absolute_error(y_test_reg, ens_pred_reg):.2f}")
print(f"Baseline:  MAE={mean_absolute_error(y_test_reg, np.full_like(y_test_reg, y_train_reg.mean())):.2f} (always predict mean)")

# ── Approach 2: Over/Under with different thresholds ──
print(f"\n=== Approach 2: Over/Under (multiple thresholds) ===")
for threshold in [20, 22, 23, 24, 25]:
    y_bin = (y_test_reg > threshold).astype(int)
    y_bin_train = (y_train_reg > threshold).astype(int)
    naive = max(y_bin.mean(), 1 - y_bin.mean())
    
    xgb_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85, random_state=42,
                             eval_metric='logloss', use_label_encoder=False)
    xgb_clf.fit(X_train, y_bin_train)
    prob = xgb_clf.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(y_bin, pred)
    lift = (acc - naive) * 100
    print(f"  O/U {threshold:2d}: acc={acc:.4f}  naive={naive:.4f}  lift={lift:+.2f}%  "
          f"(over={y_bin.mean():.1%}, under={1-y_bin.mean():.1%})")

# ── Approach 3: Over/Under by best_of ──
print(f"\n=== Approach 3: Split by best_of ===")
for bo, name in [(0, "Best-of-3"), (1, "Best-of-5")]:
    mask_train = train['is_bo5'] == bo
    mask_test = test['is_bo5'] == bo
    if mask_test.sum() < 100:
        continue
    
    X_tr = X_train[mask_train]
    y_tr_reg = y_train_reg[mask_train]
    X_te = X_test[mask_test]
    y_te_reg = y_test_reg[mask_test]
    
    threshold = int(np.median(y_tr_reg))
    y_tr_bin = (y_tr_reg > threshold).astype(int)
    y_te_bin = (y_te_reg > threshold).astype(int)
    naive = max(y_te_bin.mean(), 1 - y_te_bin.mean())
    
    xgb_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85, random_state=42,
                             eval_metric='logloss', use_label_encoder=False)
    xgb_clf.fit(X_tr, y_tr_bin)
    prob = xgb_clf.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(y_te_bin, pred)
    lift = (acc - naive) * 100
    
    print(f"  {name} (n={mask_test.sum()}, threshold={threshold}):")
    print(f"    acc={acc:.4f}  naive={naive:.4f}  lift={lift:+.2f}%")

# Feature importance
print(f"\n=== Feature Importance (XGB regression) ===")
imp = sorted(zip(TOTAL_FEATURE_NAMES, xgb_reg.feature_importances_), key=lambda x: -x[1])
for name, score in imp:
    print(f"  {name:25s} {score:.4f}")
