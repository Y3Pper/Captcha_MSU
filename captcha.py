import pandas as pd
import numpy as np

import json

from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

name_tst = 'test.parquet'
name_tr = 'train.parquet'
unlabelled = 'unlabelled.parquet'

data = pd.read_parquet(name_tr)
tst = pd.read_parquet(name_tst)
unlabel = pd.read_parquet(unlabelled)

print(data.info())
print(data['target'].value_counts())

def extract_touch_features(row):
    feats = {}
    raw = row['touch_events']

    if not isinstance(raw, str) or raw == '[]':

        for col in ['touch_cnt', 'touch_duration']:
            feats[col] = 0
        for param in ['force', 'radius_ratio', 'angle']:
            for stat in ['mean', 'std', 'min', 'max']:
                feats[f'touch_{param}_{stat}'] = 0.0
        return feats

    try:
        events = json.loads(raw)
    except:
        # битая JSON-строка
        return feats

    if not events:
        return feats

    times = np.array([e['timestamp_'] for e in events], dtype=float)
    xs = np.array([e['x_'] for e in events], dtype=float)
    ys = np.array([e['y_'] for e in events], dtype=float)
    forces = np.array([e['force_'] for e in events], dtype=float)
    rx = np.array([e['radiusX_'] for e in events], dtype=float)
    ry = np.array([e['radiusY_'] for e in events], dtype=float)
    angles = np.array([e['rotationAngle_'] for e in events], dtype=float)

    feats['touch_force_mean'] = forces.mean()
    feats['touch_force_std'] = forces.std()
    feats['touch_force_min'] = forces.min()
    feats['touch_force_max'] = forces.max()
    if len(times) > 1:
        feats['touch_force_trend'] = np.polyfit(times, forces, 1)[0]  # наклон
    else:
        feats['touch_force_trend'] = 0.0

    ratio = rx / (ry + 1e-6)
    feats['touch_radius_ratio_mean'] = ratio.mean()
    feats['touch_radius_ratio_std'] = ratio.std()
    feats['touch_radius_ratio_min'] = ratio.min()
    feats['touch_radius_ratio_max'] = ratio.max()

    feats['touch_angle_mean'] = angles.mean()
    feats['touch_angle_std'] = angles.std()
    feats['touch_angle_min'] = angles.min()
    feats['touch_angle_max'] = angles.max()
    if len(angles) > 1:
        angle_diff = np.diff(angles)
        feats['touch_angle_diff_mean'] = angle_diff.mean()
        feats['touch_angle_diff_std'] = angle_diff.std()
    else:
        feats['touch_angle_diff_mean'] = 0.0
        feats['touch_angle_diff_std'] = 0.0

    # Движение:
    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(times)
    valid = dt > 0
    if valid.sum() > 0:
        dx = dx[valid]
        dy = dy[valid]
        dt = dt[valid]
        speeds = np.sqrt(dx**2 + dy**2) / dt
        feats['touch_speed_mean'] = speeds.mean()
        feats['touch_speed_std'] = speeds.std()
        feats['touch_speed_max'] = speeds.max()
        if len(speeds) > 1:
            acc = np.diff(speeds) / dt[1:]
            feats['touch_acc_mean'] = np.abs(acc).mean()
            feats['touch_acc_std'] = acc.std()
        else:
            feats['touch_acc_mean'] = 0.0
            feats['touch_acc_std'] = 0.0
        path_length = (speeds * dt).sum()
        straight_dist = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
        feats['touch_path_length'] = path_length
        feats['touch_straightness'] = straight_dist / (path_length + 1e-5)
    else:
        # одно событие или нет движения
        for k in ['speed_mean','speed_std','speed_max','acc_mean','acc_std','path_length','straightness']:
            feats[f'touch_{k}'] = 0.0

    feats['touch_cnt'] = len(events)
    feats['touch_duration'] = times[-1] - times[0] if len(times) > 0 else 0

    return feats


def extract_mouse_features(row):
    feats = {}
    raw = row.get('mouse_events')

    if not isinstance(raw, str) or raw == '[]':
        for key in ['cnt', 'duration', 'dt_mean', 'dt_std', 'dt_min', 'dt_max', 'dt_lt20',
                    'v_mean', 'v_std', 'v_max', 'v_peak_ratio',
                    'a_mean', 'a_std',
                    'path_length', 'straight_dist', 'tortuosity',
                    'bbox_area', 'angle_mean', 'angle_std']:
            feats[f'mouse_{key}'] = 0.0
        feats['mouse_total_ratio'] = 0.0
        return feats

    try:
        events = json.loads(raw)
    except:
        return feats

    if len(events) < 2:
        feats['mouse_cnt'] = len(events)
        feats['mouse_duration'] = 0
        feats['mouse_total_ratio'] = row.get('mouse_events_total', 0) / max(len(events), 1)

        return feats

    # Извлекаем массивы
    t = np.array([e['timestamp_'] for e in events], dtype=float)
    x = np.array([e['x_'] for e in events], dtype=float)
    y = np.array([e['y_'] for e in events], dtype=float)

    dt = np.diff(t)

    valid = dt > 0
    if valid.sum() == 0:
        feats['mouse_cnt'] = len(events)
        feats['mouse_duration'] = 0
        feats['mouse_total_ratio'] = row.get('mouse_events_total', 0) / max(len(events), 1)
        return feats

    dt = dt[valid]
    dx = np.diff(x)[valid]
    dy = np.diff(y)[valid]

    feats['mouse_dt_mean'] = dt.mean()
    feats['mouse_dt_std'] = dt.std()
    feats['mouse_dt_min'] = dt.min()
    feats['mouse_dt_max'] = dt.max()
    feats['mouse_dt_lt20'] = (dt < 20).mean()  # доля быстрых интервалов

    # скорость
    speeds = np.sqrt(dx ** 2 + dy ** 2) / dt
    feats['mouse_v_mean'] = speeds.mean()
    feats['mouse_v_std'] = speeds.std()
    feats['mouse_v_max'] = speeds.max()
    feats['mouse_v_peak_ratio'] = speeds.max() / (speeds.mean() + 1e-6)

    #  ускорение
    if len(speeds) > 1:
        acc = np.diff(speeds) / dt[1:]
        feats['mouse_a_mean'] = np.abs(acc).mean()
        feats['mouse_a_std'] = acc.std()
    else:
        feats['mouse_a_mean'] = 0.0
        feats['mouse_a_std'] = 0.0

    #  геометрия
    path_length = np.sum(speeds * dt)
    straight_dist = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
    feats['mouse_path_length'] = path_length
    feats['mouse_straight_dist'] = straight_dist
    feats['mouse_tortuosity'] = path_length / (straight_dist + 1e-6)

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    feats['mouse_bbox_area'] = (max_x - min_x) * (max_y - min_y)

    vectors = np.column_stack([dx, dy])  # массив (N,2)
    norms = np.linalg.norm(vectors, axis=1)
    angles = []
    for i in range(len(vectors) - 1):
        if norms[i] > 0 and norms[i + 1] > 0:
            cos = np.dot(vectors[i], vectors[i + 1]) / (norms[i] * norms[i + 1])
            cos = np.clip(cos, -1.0, 1.0)
            angles.append(np.arccos(cos))
    if angles:
        angles = np.array(angles)
        feats['mouse_angle_mean'] = angles.mean()
        feats['mouse_angle_std'] = angles.std()
    else:
        feats['mouse_angle_mean'] = 0.0
        feats['mouse_angle_std'] = 0.0

    # общее
    feats['mouse_cnt'] = len(events)
    feats['mouse_duration'] = t[-1] - t[0]
    feats['mouse_total_ratio'] = row.get('mouse_events_total', 0) / max(len(events), 1)

    return feats



tst_touch = tst.apply(extract_touch_features, axis=1, result_type='expand')
data_touch = data.apply(extract_touch_features, axis=1, result_type='expand')
unlabel_touch = unlabel.apply(extract_touch_features, axis=1, result_type='expand')

tst_mouse = tst.apply(extract_mouse_features, axis=1, result_type='expand')
data_mouse = data.apply(extract_mouse_features, axis=1, result_type='expand')
unlabel_mouse = unlabel.apply(extract_mouse_features, axis=1, result_type='expand')

tst = pd.concat([tst, tst_touch, tst_mouse], axis=1)
data = pd.concat([data, data_touch, data_mouse], axis=1)
unlabel = pd.concat([unlabel, unlabel_touch, unlabel_mouse], axis=1)

tst = tst.drop(['mouse_events', 'touch_events'], axis=1)
data = data.drop(['mouse_events', 'touch_events'], axis=1)
unlabel = unlabel.drop(['mouse_events', 'touch_events'], axis=1)

X_tr = data.drop(['target'], axis=1)
y_tr = data['target']


isof = IsolationForest(n_estimators = 500, contamination=0.2, random_state=42)
data_isof = pd.concat([X_tr, unlabel], axis=0)
isof.fit(data_isof)
X_tr['anomality'] = isof.decision_function(X_tr)
tst['anomality'] = isof.decision_function(tst)


X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr)


cb = CatBoostClassifier(random_seed=42,
                        learning_rate=0.03,
                        iterations=2000,
                        depth=6,
                        l2_leaf_reg=3,
                        eval_metric='AUC',
                        early_stopping_rounds=200,
                        verbose=100)

xgb = XGBClassifier(    random_seed = 42,
                        n_estimators = 1000,
                        learning_rate = 0.05,
                        max_depth = 5,
                        eval_metric = 'auc')

lgb = LGBMClassifier(   random_seed=42,
                        n_estimators=1000,
                        learning_rate = 0.05,
                        max_depth = 5,
                        num_leaves=30,
                        verbose=-1)

cb.fit(X_train, y_train, eval_set=(X_val, y_val))
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)

p_cb = cb.predict_proba(X_val)[:, 1]
p_lgb = lgb.predict_proba(X_val)[:, 1]
p_xgb = xgb.predict_proba(X_val)[:, 1]

w = [0.5, 0.25, 0.25]
p_m = 0
for w_cb in np.arange(0.0, 0.9, 0.05):
    w_rest = (1-w_cb)/2
    p_w = (w_cb*p_cb + w_rest*(p_lgb+p_xgb))
    p_auc = roc_auc_score(y_val, p_w, max_fpr=0.1)
    if p_auc > p_m:
        p_m = p_auc
        w = [w_cb, w_rest, w_rest]
print(f"Best weights (cat, lgb, xgb): {w}, p_AUC={p_m}")


cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
xgb.fit(X_tr, y_tr)
lgb.fit(X_tr, y_tr)

p_cb = cb.predict_proba(tst)[:, 1]
p_lgb = lgb.predict_proba(tst)[:, 1]
p_xgb = xgb.predict_proba(tst)[:, 1]

p = w[0]*p_cb+w[1]*p_lgb+w[2]*p_xgb


subm = pd.DataFrame({
    'id': range(len(p)),
    'prediction': p})

subm.to_parquet('submission.parquet', index=False)