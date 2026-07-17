"""Tennis total games over/under model."""
import sys, re
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tennis.tennis_db as tennis_db
from tennis.features import TennisFeatureEngine, FEATURE_NAMES

# Step 6-7: Parse total games and find thresholds
print("Step 6-7: Parsing total games from scores...")

all_scores = []
with tennis_db.connect() as conn:
    rows = [tuple(r) for r in conn.execute("""
        SELECT m.player1_id, m.player2_id, m.winner_id, m.surface, m.date, m.score,
               m.w_1stIn, m.w_svpt, m.w_1stWon, m.w_2ndWon, m.w_bpSaved, m.w_bpFaced,
               m.l_1stIn, m.l_svpt, m.l_1stWon, m.l_2ndWon, m.l_bpSaved, m.l_bpFaced,
               p1.ranking, p2.ranking
        FROM tennis_matches m
        LEFT JOIN tennis_players p1 ON m.player1_id = p1.id
        LEFT JOIN tennis_players p2 ON m.player2_id = p2.id
        WHERE m.status='finished' AND m.date IS NOT NULL AND m.score IS NOT NULL
        ORDER BY m.date ASC
    """).fetchall()]

valid = 0
invalid = 0
total_games_list = []
bo3_games = []
bo5_games = []

for r in rows:
    score = r[5]
    if not score:
        invalid += 1
        continue
    # Skip walkovers, retirements
    if any(x in str(score).upper() for x in ['W/O', 'RET', 'DEF', 'ABN', 'UNF', 'W/O']):
        invalid += 1
        continue
    
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        invalid += 1
        continue
    
    total = sum(int(a) + int(b) for a, b in sets)
    n_sets = len(sets)
    
    # Determine best_of from number of sets
    # If max sets = 3 -> best_of=3, if max sets = 5 -> best_of=5
    # Heuristic: if any set score > 7, it's likely a tiebreak in best_of=5
    max_game = max(max(int(a), int(b)) for a, b in sets)
    
    total_games_list.append(total)
    if n_sets <= 2:
        bo3_games.append(total)
    else:
        bo5_games.append(total)
    valid += 1

print(f"Valid scores: {valid}, Invalid: {invalid}")

# Step 7: Distribution analysis
print(f"\n=== Total Games Distribution ===")
print(f"Overall: median={np.median(total_games_list):.0f}, mean={np.mean(total_games_list):.1f}, "
      f"Q25={np.percentile(total_games_list, 25):.0f}, Q75={np.percentile(total_games_list, 75):.0f}")

print(f"\nBest-of-3 (n={len(bo3_games)}):")
print(f"  median={np.median(bo3_games):.0f}, mean={np.mean(bo3_games):.1f}, "
      f"Q25={np.percentile(bo3_games, 25):.0f}, Q75={np.percentile(bo3_games, 75):.0f}")

print(f"\nBest-of-5 (n={len(bo5_games)}):")
print(f"  median={np.median(bo5_games):.0f}, mean={np.mean(bo5_games):.1f}, "
      f"Q25={np.percentile(bo5_games, 25):.0f}, Q75={np.percentile(bo5_games, 75):.0f}")

# Thresholds
bo3_median = np.median(bo3_games)
bo5_median = np.median(bo5_games)
print(f"\nRecommended thresholds:")
print(f"  Best-of-3: O/U {bo3_median:.0f} (median)")
print(f"  Best-of-5: O/U {bo5_median:.0f} (median)")
print(f"  Combined:  O/U {np.median(total_games_list):.0f} (median)")

# Step 8: Build dataset and train
print(f"\n=== Step 8: Training Total Games Model ===")

engine = TennisFeatureEngine()
total_rows = []

# Debug: check row length
if rows:
    print(f"First row length: {len(rows[0])}")

for r in rows:
    if len(r) < 20:
        continue  # skip malformed rows
    p1_id, p2_id, winner_id = r[0], r[1], r[2]
    surface, date, score = r[3] or 'hard', r[4], r[5]
    
    if not score:
        continue
    if any(x in str(score).upper() for x in ['W/O', 'RET', 'DEF', 'ABN', 'UNF']):
        continue
    
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        continue
    
    total = sum(int(a) + int(b) for a, b in sets)
    n_sets = len(sets)
    is_bo5 = 1 if n_sets > 2 else 0
    
    p1_rank, p2_rank = r[18], r[19]
    
    # Get features BEFORE updating
    feats = engine.get_features(p1_id, p2_id, surface, p1_rank, p2_rank, date)
    
    # Label: over median (1) or under (0)
    threshold = bo5_median if is_bo5 else bo3_median
    label = 1 if total > threshold else 0
    
    total_rows.append({**feats, 'label': label, 'total_games': total, 
                       'threshold': threshold, 'is_bo5': is_bo5, 'date': date})
    
    # Update engine
    if winner_id == p1_id:
        w_id, l_id = p1_id, p2_id
    else:
        w_id, l_id = p2_id, p1_id
    engine.update(w_id, l_id, surface, date)

df = pd.DataFrame(total_rows)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=FEATURE_NAMES + ['date'])

# Add best_of as feature
df['is_bo5'] = df['is_bo5'].astype(int)

print(f"Dataset: {len(df)} rows")
print(f"Over/Under distribution: {df['label'].value_counts().to_dict()}")
print(f"Naive baseline (majority class): {max(df['label'].mean(), 1-df['label'].mean()):.4f}")

# Split
split_date = df['date'].quantile(0.8)
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
print(f"Train: {len(train)}, Test: {len(test)}")

feature_cols = FEATURE_NAMES + ['is_bo5']
X_train = train[feature_cols].values
y_train = train['label'].values
X_test = test[feature_cols].values
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

naive_acc = max(y_test.mean(), 1 - y_test.mean())

print(f"\n=== Results ===")
print(f"Naive baseline:  acc={naive_acc:.4f}")
print(f"XGBoost:         acc={accuracy_score(y_test, xgb_pred):.4f}  ll={log_loss(y_test, xgb_prob):.4f}")
print(f"LightGBM:        acc={accuracy_score(y_test, lgbm_pred):.4f}  ll={log_loss(y_test, lgbm_prob):.4f}")
print(f"Ensemble:        acc={accuracy_score(y_test, ens_pred):.4f}  ll={log_loss(y_test, ens_prob):.4f}")
print(f"\nLift over naive: {(accuracy_score(y_test, ens_pred) - naive_acc)*100:+.2f}%")

print(f"\nTop features:")
imp = sorted(zip(feature_cols, xgb.feature_importances_), key=lambda x: -x[1])
for name, score in imp[:8]:
    print(f"  {name:25s} {score:.4f}")
