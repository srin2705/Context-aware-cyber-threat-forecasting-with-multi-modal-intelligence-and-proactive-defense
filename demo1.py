"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     CONTEXT-AWARE CYBER THREAT FORECASTING — FYP FINAL DEMO               ║
║     XGBoost + LSTM (MC Dropout) + Adaptive Markov v3                       ║
║                                                                             ║
║  Pipeline:  XGBoost (calibrated) → LSTM (MC Dropout) → Adaptive Markov v3  ║
║  Goal:      Observe current network traffic → Forecast NEXT threat class    ║
║  Modes:     1=Scenario Sweep  2=Real Samples  3=Interactive  4=Stress Test  ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python FYP_FINAL_DEMO.py

REQUIREMENTS:
    Place this file in the same folder as  fyp_saved_models/
    The folder must contain:
        xgb_calibrated.pkl   lstm_model.keras    label_encoder.pkl
        X_bal.npy            y_bal.npy           class_names.json
        feature_cols.json    scaler.pkl
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os, sys, json, warnings, datetime, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import joblib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['font.size'] = 11
sns.set_style('darkgrid')

CLASS_COLORS = {
    'DDoS':           '#e74c3c',
    'DoS':            '#e67e22',
    'Reconnaissance': '#3498db',
    'Normal':         '#2ecc71',
}
ALERT_COLORS = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'}
ALERT_ICON   = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟢'}
CLASS_ICON   = {'DDoS': '💥', 'DoS': '⚡', 'Reconnaissance': '🔍', 'Normal': '✅'}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD SAVED MODELS & ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fyp_saved_models')
WINDOW    = 10
THRESHOLD = 0.70

def load_artifacts():
    print("=" * 65)
    print("  LOADING SAVED MODELS & ARTIFACTS")
    print("=" * 65)

    required = [
        'xgb_calibrated.pkl', 'lstm_model.keras',
        'label_encoder.pkl',  'X_bal.npy',
        'y_bal.npy',          'class_names.json',
        'feature_cols.json',
    ]
    missing = [f for f in required if not os.path.exists(f'{SAVE_DIR}/{f}')]
    if missing:
        print(f"\n  ❌ Missing files in '{SAVE_DIR}/': {missing}")
        print("  Place fyp_saved_models/ in the same folder as this script.")
        sys.exit(1)

    cal_xgb    = joblib.load(f'{SAVE_DIR}/xgb_calibrated.pkl')
    lstm_model = tf.keras.models.load_model(f'{SAVE_DIR}/lstm_model.keras')
    le         = joblib.load(f'{SAVE_DIR}/label_encoder.pkl')
    X_bal      = np.load(f'{SAVE_DIR}/X_bal.npy')
    y_bal      = np.load(f'{SAVE_DIR}/y_bal.npy')

    with open(f'{SAVE_DIR}/class_names.json')  as f: class_names = np.array(json.load(f))
    with open(f'{SAVE_DIR}/feature_cols.json') as f: features    = json.load(f)

    print(f"  ✅ XGBoost (calibrated)   loaded")
    print(f"  ✅ LSTM model             loaded")
    print(f"  ✅ LabelEncoder           loaded  → classes: {list(class_names)}")
    print(f"  ✅ X_bal / y_bal          loaded  → shape: {X_bal.shape}")
    print(f"  ✅ Feature list           loaded  → {len(features)} features")
    print()

    return cal_xgb, lstm_model, le, X_bal, y_bal, class_names, features


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STOCHASTIC MODULES
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Markov Transition Matrices ────────────────────────────────────────────
def build_markov_matrix(labels, n_cls):
    """Empirical P(next=j | current=i) from label sequence."""
    M = np.zeros((n_cls, n_cls))
    for a, b in zip(labels[:-1], labels[1:]):
        M[int(a), int(b)] += 1
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return M / rs

def build_transition_only_markov(markov_matrix):
    """Zeros diagonal — answers: IF a change happens, to what class?"""
    m = markov_matrix.copy()
    np.fill_diagonal(m, 0)
    rs = m.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return m / rs

def build_escalation_prior(class_names):
    """
    Domain knowledge: cyber kill-chain escalation priors.
    Normal → Recon → DoS → DDoS (attacker progression).
    """
    n   = len(class_names)
    idx = {str(name): i for i, name in enumerate(class_names)}
    ep  = np.zeros((n, n))

    if 'Normal' in idx and 'Reconnaissance' in idx:
        ep[idx['Normal']][idx['Reconnaissance']] = 0.50
        ep[idx['Normal']][idx['DoS']]            = 0.20
        ep[idx['Normal']][idx['DDoS']]           = 0.10
        ep[idx['Normal']][idx['Normal']]         = 0.20

    if 'Reconnaissance' in idx:
        ep[idx['Reconnaissance']][idx['DoS']]            = 0.40
        ep[idx['Reconnaissance']][idx['DDoS']]           = 0.30
        ep[idx['Reconnaissance']][idx['Reconnaissance']] = 0.20
        ep[idx['Reconnaissance']][idx['Normal']]         = 0.10

    if 'DoS' in idx:
        ep[idx['DoS']][idx['DDoS']]           = 0.45
        ep[idx['DoS']][idx['DoS']]            = 0.30
        ep[idx['DoS']][idx['Reconnaissance']] = 0.10
        ep[idx['DoS']][idx['Normal']]         = 0.15

    if 'DDoS' in idx:
        ep[idx['DDoS']][idx['DDoS']]           = 0.60
        ep[idx['DDoS']][idx['DoS']]            = 0.20
        ep[idx['DDoS']][idx['Normal']]         = 0.20

    rs = ep.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return ep / rs


# ── 2b. MC Dropout for LSTM ───────────────────────────────────────────────────
MC_DROPOUT_PASSES          = 30
MC_DROPOUT_RATE            = 0.20
UNCERTAINTY_WEIGHT         = 0.15
UNCERTAINTY_HIGH_THRESHOLD = 0.60

def setup_mc_dropout(lstm_model):
    """Patch LSTM with MC Dropout for uncertainty estimation."""
    has_dropout = any(isinstance(l, tf.keras.layers.Dropout) for l in lstm_model.layers)
    if has_dropout:
        print("  [MC Dropout] Model has Dropout layers → using training=True")
        def mc_forward(x):
            return lstm_model(x, training=True).numpy()[0]
    else:
        print(f"  [MC Dropout] No Dropout found → patching with {MC_DROPOUT_RATE:.0%} wrapper")
        inp      = lstm_model.input
        dropped  = tf.keras.layers.Dropout(MC_DROPOUT_RATE)(lstm_model.output)
        mc_model = tf.keras.Model(inputs=inp, outputs=dropped)
        def mc_forward(x):
            return mc_model(x, training=True).numpy()[0]
    return mc_forward

def lstm_predict_mc(lstm_inp, mc_forward, n_passes=MC_DROPOUT_PASSES):
    """N stochastic forward passes → mean probabilities + normalised entropy + std."""
    preds      = np.array([mc_forward(lstm_inp) for _ in range(n_passes)])
    mean_proba = preds.mean(axis=0)
    std_proba  = preds.std(axis=0)
    eps        = 1e-12
    entropy    = -np.sum(mean_proba * np.log(mean_proba + eps))
    norm_unc   = float(entropy / np.log(len(mean_proba))) if len(mean_proba) > 1 else 0.0
    return mean_proba, norm_unc, std_proba


# ── 2c. Context Engine ─────────────────────────────────────────────────────────
CONTEXT_WEIGHTS = {
    'time': 0.15, 'device': 0.25, 'network': 0.30,
    'threat_history': 0.20, 'geolocation': 0.10,
}

def context_time(hour=None):
    if hour is None: hour = datetime.datetime.now().hour
    return 0.2 if 8 <= hour <= 18 else (0.6 if 19 <= hour <= 23 else 0.9)

def context_device(device_type='unknown'):
    return {'camera': 0.8, 'sensor': 0.7, 'workstation': 0.3,
            'server': 0.2, 'unknown': 0.5}.get(device_type, 0.5)

def context_network(pkts_per_sec=0.0, unique_dsts=1):
    return min(pkts_per_sec / 1000.0, 1.0) * 0.6 + min(unique_dsts / 50.0, 1.0) * 0.4

def context_threat_history(repeat_count=0):
    return [0.1, 0.5, 0.75, 0.75, 0.95][min(repeat_count, 4)]

def context_geolocation(country_code='US'):
    high   = {'CN', 'RU', 'KP', 'IR', 'SY'}
    medium = {'NG', 'UA', 'RO', 'BR', 'IN', 'PK', 'VN', 'TH', 'ID',
              'EG', 'TR', 'MX', 'ZA', 'BD', 'PH'}
    low    = {'US', 'GB', 'DE', 'FR', 'CA', 'AU', 'NL', 'JP', 'SE',
              'NO', 'CH', 'NZ', 'SG', 'IE', 'FI'}
    code = country_code.upper().strip()
    return 0.90 if code in high else (0.55 if code in medium else (0.15 if code in low else 0.50))

def compute_context_score(hour=None, device_type='unknown', pkts_per_sec=0.0,
                           unique_dsts=1, repeat_count=0, country_code='US'):
    dims = {
        'time'          : context_time(hour),
        'device'        : context_device(device_type),
        'network'       : context_network(pkts_per_sec, unique_dsts),
        'threat_history': context_threat_history(repeat_count),
        'geolocation'   : context_geolocation(country_code),
    }
    score = np.clip(sum(dims[k] * CONTEXT_WEIGHTS[k] for k in CONTEXT_WEIGHTS), 0.0, 1.0)
    return score, dims


# ── 2d. Proactive Defence Actions ─────────────────────────────────────────────
PROACTIVE_ACTIONS = {
    'DDoS':           ['🚫 BLOCK IP', '⚡ RATE LIMIT', '📢 ALERT ADMIN', '🛡 ACTIVATE DDoS MITIGATION'],
    'DoS':            ['🚫 BLOCK IP', '🔒 ISOLATE DEVICE', '📢 ALERT ADMIN'],
    'Reconnaissance': ['📝 LOG SCAN', '👁 INCREASE MONITORING', '🔥 UPDATE FIREWALL RULES'],
    'Normal':         ['✅ CONTINUE MONITORING'],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SYNTHETIC INPUT GENERATOR (from saved X_bal data)
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_pool(X_bal, y_bal, class_names, le):
    """Build per-class sample pools and noise scales from X_bal."""
    pool        = {}
    noise_scale = {}
    for cls in class_names:
        cls_id            = le.transform([cls])[0]
        idx               = np.where(y_bal == cls_id)[0]
        np.random.seed(42); np.random.shuffle(idx)
        pool[cls]         = X_bal[idx[:50]]
        noise_scale[cls]  = X_bal[y_bal == cls_id].std(axis=0) * 0.05
    return pool, noise_scale

def simulate_window(cls_name, step_idx, pool, noise_scale, cal_xgb):
    """Generate a stochastic XGBoost probability vector with 5% noise."""
    p       = pool[cls_name]
    indices = [(step_idx + i) % len(p) for i in range(5)]
    feat    = np.mean(p[indices], axis=0)
    feat    = feat + np.random.normal(0, noise_scale[cls_name], feat.shape)
    return cal_xgb.predict_proba(feat.reshape(1, -1))[0]

def real_sample_window(cls_name, pool, noise_scale, cal_xgb):
    """Pick one RANDOM real sample from the class pool (for Mode 2)."""
    p    = pool[cls_name]
    feat = p[np.random.randint(len(p))].copy()
    feat = feat + np.random.normal(0, noise_scale[cls_name] * 0.5, feat.shape)
    return cal_xgb.predict_proba(feat.reshape(1, -1))[0]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DECISION ENGINE v3 (Next-State Forecasting)
# ══════════════════════════════════════════════════════════════════════════════

def detect_xgb_trend(xgb_history, n_cls):
    """
    Detect emerging class shifts by comparing the latest XGB output
    against the running average.  Returns divergence [0,1] and direction.
    """
    if len(xgb_history) < 3:
        return 0.0, np.ones(n_cls) / n_cls
    recent   = np.array(list(xgb_history))
    prev_avg = recent[:-1].mean(axis=0)
    current  = recent[-1]
    div      = min(float(np.abs(current - prev_avg).sum() / 2.0), 1.0)
    delta    = current - prev_avg
    pos      = np.maximum(delta, 0)
    return div, (pos / pos.sum() if pos.sum() > 1e-8 else np.ones(n_cls) / n_cls)

def markov_lookahead(current_class, class_names, markov_matrix, n_steps=3):
    """Multi-step Markov lookahead: probabilities at t+1, t+2, t+3."""
    idx  = list(class_names).index(current_class)
    dist = np.zeros(len(class_names)); dist[idx] = 1.0
    forecasts = []
    for _ in range(n_steps):
        dist = dist @ markov_matrix
        forecasts.append({str(class_names[i]): round(float(dist[i]), 4)
                          for i in range(len(class_names))})
    return forecasts

def decision_engine_v3(xgb_proba, lstm_proba, current_state_idx, step_idx,
                        markov_matrix, transition_only_markov, escalation_prior,
                        n_classes, class_names,
                        context_kwargs=None, uncertainty=0.0,
                        state_history=None, xgb_history=None):
    """
    NEXT-STATE FORECASTING ENGINE v3
    ─────────────────────────────────
    At time t: observes current class → forecasts the class at t+1.

    Fusion weights: XGB=5%  LSTM=15%  Adaptive Markov=80%

    Adaptive Markov blends three matrices:
      raw_markov           — empirical data transitions (stability)
      transition_only      — transitions when change IS happening
      escalation_prior     — domain kill-chain knowledge

    Blending is controlled by context_score × 1.5, boosted by
    state volatility and XGB trend divergence.
    """
    if context_kwargs is None: context_kwargs = {}
    if state_history  is None: state_history  = []
    if xgb_history    is None: xgb_history    = deque()

    ctx_score, ctx_dims = compute_context_score(**context_kwargs)
    trend_div, _        = detect_xgb_trend(xgb_history, n_classes)

    # Adaptive Markov blending
    raw_m   = markov_matrix[current_state_idx]
    trans_m = transition_only_markov[current_state_idx]
    esc_p   = escalation_prior[current_state_idx]

    ctx_blend = min(ctx_score * 1.5, 0.90)

    if len(state_history) >= 3:
        changes    = sum(1 for i in range(1, len(state_history))
                         if state_history[i] != state_history[i-1])
        volatility = changes / (len(state_history) - 1)
        if volatility > 0.3:
            ctx_blend = min(ctx_blend + 0.15, 0.95)

    ctx_blend  = min(ctx_blend + trend_div * 0.1, 0.95)
    stab_w     = 1.0 - ctx_blend
    adaptive_m = (stab_w * raw_m +
                  ctx_blend * 0.4 * trans_m +
                  ctx_blend * 0.6 * esc_p)
    adaptive_m = adaptive_m / (adaptive_m.sum() + 1e-12)

    fused      = 0.05 * xgb_proba + 0.15 * lstm_proba + 0.80 * adaptive_m
    next_idx   = int(np.argmax(fused))
    next_cls   = str(class_names[next_idx])
    next_conf  = float(fused[next_idx])

    unc_adj    = uncertainty * UNCERTAINTY_WEIGHT
    final_risk = np.clip(0.65 * next_conf + 0.35 * ctx_score + unc_adj, 0.0, 1.0)

    if final_risk >= THRESHOLD:  alert = 'HIGH'
    elif final_risk >= 0.45:     alert = 'MEDIUM'
    else:                        alert = 'LOW'

    unc_flag = 'HIGH' if uncertainty >= UNCERTAINTY_HIGH_THRESHOLD else 'LOW'
    if unc_flag == 'HIGH' and alert == 'HIGH':
        alert = 'MEDIUM'

    actions = PROACTIVE_ACTIONS.get(next_cls, ['CONTINUE MONITORING'])
    if unc_flag == 'HIGH':
        actions = ['⚠ VERIFY — HIGH UNCERTAINTY'] + actions

    return {
        'step'                 : step_idx,
        'predicted_next_class' : next_cls,
        'model_confidence'     : round(next_conf, 4),
        'context_score'        : round(ctx_score, 4),
        'context_dims'         : {k: round(v, 3) for k, v in ctx_dims.items()},
        'uncertainty'          : round(uncertainty, 4),
        'uncertainty_flag'     : unc_flag,
        'final_risk'           : round(final_risk, 4),
        'alert_level'          : alert,
        'actions'              : actions,
        'transition_likelihood': round(ctx_blend, 4),
        'xgb_trend_divergence' : round(trend_div, 4),
        'fused_proba'          : np.round(fused, 4).tolist(),
        'adaptive_markov'      : np.round(adaptive_m, 4).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CORE SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(name, sequence, ctx, cal_xgb, lstm_model, le, pool, noise_scale,
                 markov_matrix, transition_only_markov, escalation_prior,
                 n_classes, class_names, mc_forward,
                 use_real_samples=False):
    """Run one scenario end-to-end. Returns list of per-step result dicts."""
    xgb_history   = deque(maxlen=WINDOW)
    lstm_history  = deque(maxlen=WINDOW)
    state_history = []
    results       = []

    for step, cls_name in enumerate(sequence):
        state_history.append(cls_name)

        if use_real_samples:
            xgb_proba = real_sample_window(cls_name, pool, noise_scale, cal_xgb)
        else:
            xgb_proba = simulate_window(cls_name, step, pool, noise_scale, cal_xgb)

        xgb_history.append(xgb_proba)
        lstm_history.append(xgb_proba)

        if len(lstm_history) >= WINDOW:
            lstm_inp               = np.array(list(lstm_history)).reshape(1, WINDOW, n_classes)
            lstm_proba, uncertainty, _ = lstm_predict_mc(lstm_inp, mc_forward)
        else:
            lstm_proba  = np.ones(n_classes) / n_classes
            uncertainty = 0.0

        csi    = int(le.transform([cls_name])[0])
        result = decision_engine_v3(
            xgb_proba, lstm_proba, csi, step,
            markov_matrix, transition_only_markov, escalation_prior,
            n_classes, class_names,
            context_kwargs=ctx, uncertainty=uncertainty,
            state_history=state_history, xgb_history=xgb_history,
        )

        true_next = sequence[step + 1] if step < len(sequence) - 1 else '(FORECAST)'
        result['true_next_class'] = true_next
        result['current_class']   = cls_name
        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PRETTY PRINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_scenario_header(name, sequence, ctx):
    print(f"\n{'═'*72}")
    print(f"  SCENARIO  : {name}")
    print(f"  Sequence  : {sequence}")
    print(f"  Context   : device={ctx['device_type']}  hour={ctx['hour']}h  "
          f"pkts/s={ctx['pkts_per_sec']}  country={ctx['country_code']}")
    print(f"{'═'*72}")
    print(f"  {'t':<4} {'Current':<16} {'→ Forecast':<16} {'True Next':<16} "
          f"{'Risk':<7} {'Alert':<10} {'Unc':<6}")
    print(f"  {'─'*3} {'─'*15} {'─'*15} {'─'*15} {'─'*6} {'─'*9} {'─'*5}")

def print_step(r, class_names, markov_matrix, show_lookahead=False):
    alert_str = f"{ALERT_ICON[r['alert_level']]} {r['alert_level']}"
    curr_icon = CLASS_ICON.get(r['current_class'], '')
    pred_icon = CLASS_ICON.get(r['predicted_next_class'], '')

    ok = ''
    if r['true_next_class'] != '(FORECAST)':
        ok = '✅' if r['predicted_next_class'] == r['true_next_class'] else '❌'

    print(f"  t{r['step']:<3} "
          f"{curr_icon+r['current_class']:<16} "
          f"→ {pred_icon+r['predicted_next_class']:<15} "
          f"{r['true_next_class']:<16} "
          f"{r['final_risk']:.3f}  "
          f"{alert_str:<16} "
          f"{r['uncertainty']:.3f} {ok}")

    if show_lookahead and r['predicted_next_class'] != 'Normal':
        forecasts = markov_lookahead(r['predicted_next_class'], class_names, markov_matrix, n_steps=3)
        top_attacks = []
        for offset, dist in enumerate(forecasts, 1):
            attack_dist = {k: v for k, v in dist.items() if k != 'Normal'}
            if attack_dist:
                top = max(attack_dist, key=attack_dist.get)
                top_attacks.append(f"{top}@t+{offset}={attack_dist[top]:.2f}")
        if top_attacks:
            print(f"  {'':4} ⚡ Lookahead: {' | '.join(top_attacks)}")

def print_defense_actions(result):
    print(f"\n  🛡  DEFENSE ACTIONS for step t{result['step']} "
          f"(predicted: {result['predicted_next_class']}):")
    for action in result['actions']:
        print(f"      → {action}")

def print_scenario_summary(name, results, class_names):
    valid  = [r for r in results if r['true_next_class'] != '(FORECAST)']
    n_corr = sum(1 for r in valid if r['predicted_next_class'] == r['true_next_class'])
    acc    = n_corr / len(valid) * 100 if valid else 0
    n_high = sum(1 for r in results if r['alert_level'] == 'HIGH')
    n_med  = sum(1 for r in results if r['alert_level'] == 'MEDIUM')
    avg_r  = np.mean([r['final_risk'] for r in results])
    avg_u  = np.mean([r['uncertainty'] for r in results])

    print(f"\n  📊 Summary — {name}")
    print(f"     Forecast Accuracy : {n_corr}/{len(valid)} = {acc:.0f}%")
    print(f"     HIGH Alerts       : {n_high}   MEDIUM: {n_med}")
    print(f"     Avg Risk Score    : {avg_r:.3f}   Avg Uncertainty: {avg_u:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALIZATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_scenario_dashboard(scenario_name, results, class_names, markov_matrix):
    """4-panel dashboard for one scenario (next-state forecasting view)."""
    n = len(results)
    steps = list(range(n))

    risks       = [r['final_risk'] for r in results]
    confs       = [r['model_confidence'] for r in results]
    ctx_scores  = [r['context_score'] for r in results]
    uncerts     = [r['uncertainty'] for r in results]
    current_cls = [r['current_class'] for r in results]
    pred_next   = [r['predicted_next_class'] for r in results]
    true_next   = [r['true_next_class'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Scenario: {scenario_name}  (Next-State Forecasting + MC Dropout)',
                 fontsize=14, fontweight='bold')

    # Panel 1: Final risk over time
    ax = axes[0, 0]
    bar_colors = ['#e74c3c' if r >= THRESHOLD else '#f39c12' if r >= 0.45 else '#2ecc71' for r in risks]
    ax.bar(steps, risks, color=bar_colors)
    ax.axhline(THRESHOLD, color='red',    ls='--', label=f'HIGH ({THRESHOLD})')
    ax.axhline(0.45,      color='orange', ls='--', label='MEDIUM (0.45)')
    ax.set_title('Forecasted Next-Step Risk', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Risk Score')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # Panel 2: Model confidence vs context score + MC uncertainty band
    ax = axes[0, 1]
    ax.plot(steps, confs,      'o-', color='#e67e22', lw=2, label='Model Confidence')
    ax.plot(steps, ctx_scores, 's-', color='#3498db', lw=1.5, label='Context Score')
    # Uncertainty as a secondary axis
    ax2 = ax.twinx()
    ax2.bar(steps, uncerts, alpha=0.25, color='#e74c3c', label='MC Uncertainty')
    ax2.axhline(UNCERTAINTY_HIGH_THRESHOLD, color='#e74c3c', ls=':', alpha=0.7)
    ax2.set_ylabel('MC Dropout Entropy', color='#e74c3c', fontsize=9)
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8)
    ax.set_title('Confidence vs Context + MC Uncertainty', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)

    # Panel 3: Markov multi-step lookahead for the final predicted class
    ax = axes[1, 0]
    final_pred = pred_next[-1]
    forecasts  = markov_lookahead(final_pred, class_names, markov_matrix, n_steps=4)
    x_labels   = [f't+{i+1}' for i in range(len(forecasts))]
    bottom_arr = np.zeros(len(forecasts))
    for cls in class_names:
        vals = [fc.get(str(cls), 0) for fc in forecasts]
        ax.bar(x_labels, vals, bottom=bottom_arr,
               label=str(cls), color=CLASS_COLORS.get(str(cls), '#aaa'), alpha=0.85)
        bottom_arr += np.array(vals)
    ax.set_title(f'Markov Lookahead from Final Predicted: {final_pred}', fontweight='bold')
    ax.set_xlabel('Future Step'); ax.set_ylabel('Class Probability')
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05)

    # Panel 4: Prediction correctness timeline
    ax = axes[1, 1]
    for i in range(n):
        is_fc = (true_next[i] == '(FORECAST)')
        if is_fc:
            color = '#9b59b6'
            label = f'Now: {current_cls[i]}  →  Pred: {pred_next[i]}  🔮'
        else:
            correct = (pred_next[i] == true_next[i])
            color   = '#2ecc71' if correct else '#e74c3c'
            label   = f'Now: {current_cls[i]}  →  Pred: {pred_next[i]}  | True: {true_next[i]}'
        ax.barh(i, 1, color=color, edgecolor='white', height=0.45)
        ax.text(0.5, i, label, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_yticks(steps)
    ax.set_yticklabels([f'Step {i}' for i in steps])
    ax.set_title('Predicted Next Class vs True Next Class', fontweight='bold')
    ax.set_xlim(0, 1); ax.invert_yaxis()

    plt.tight_layout()
    fname = f"dashboard_{scenario_name}.png"
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  📊 Dashboard saved: {fname}")


def plot_markov_matrices(markov_matrix, transition_only_markov, escalation_prior, class_names):
    """Visualise all three Markov matrices as heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, mat, title in zip(axes,
        [markov_matrix, transition_only_markov, escalation_prior],
        ['Raw Markov P(next|current)',
         'Transition-Only Markov (diagonal zeroed)',
         'Attack Escalation Prior (domain knowledge)']):
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, annot_kws={'size': 10}, ax=ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Next State'); ax.set_ylabel('Current State')
    plt.tight_layout()
    plt.savefig('markov_matrices.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("  📊 Markov matrices saved: markov_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SCENARIO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'A_AllNormal': {
        'sequence': ['Normal'] * 10,
        'ctx': dict(hour=10, device_type='workstation', pkts_per_sec=100,
                    unique_dsts=5, repeat_count=0, country_code='US'),
        'desc': 'Baseline — pure normal traffic, low-risk US workstation at daytime',
    },
    'B_SlowEscalation': {
        'sequence': ['Normal','Normal','Normal','Reconnaissance','Reconnaissance',
                     'DoS','DoS','DDoS','DDoS','DDoS'],
        'ctx': dict(hour=22, device_type='camera', pkts_per_sec=500,
                    unique_dsts=30, repeat_count=2, country_code='CN'),
        'desc': 'Classic kill-chain: Normal → Recon → DoS → DDoS (CN camera, night)',
    },
    'C_SuddenDDoS': {
        'sequence': ['Normal','Normal','DDoS','DDoS','DDoS','DDoS','DDoS','DDoS','DDoS','Normal'],
        'ctx': dict(hour=3, device_type='server', pkts_per_sec=950,
                    unique_dsts=45, repeat_count=3, country_code='RU'),
        'desc': 'No warning — DDoS bursts suddenly from a RU server at 3 AM',
    },
    'D_StealthRecon': {
        'sequence': ['Normal','Reconnaissance','Normal','Reconnaissance','Normal',
                     'Reconnaissance','Normal','Normal','Normal','Normal'],
        'ctx': dict(hour=14, device_type='sensor', pkts_per_sec=200,
                    unique_dsts=15, repeat_count=1, country_code='UA'),
        'desc': 'Stealthy attacker — alternates Normal/Recon to avoid detection (UA sensor)',
    },
    'E_APTSimulation': {
        'sequence': ['Normal','Normal','Reconnaissance','Normal','Reconnaissance',
                     'Reconnaissance','DoS','Normal','DoS','DDoS'],
        'ctx': dict(hour=1, device_type='server', pkts_per_sec=700,
                    unique_dsts=38, repeat_count=3, country_code='KP'),
        'desc': 'APT-style: slow recon with deceptive Normal gaps → burst attack (KP server, 1 AM)',
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SIMULATION MODES
# ══════════════════════════════════════════════════════════════════════════════

# ── Mode 1: Standard Scenario Sweep ───────────────────────────────────────────
def demo_scenarios(cal_xgb, lstm_model, le, pool, noise_scale, markov_matrix,
                   transition_only_markov, escalation_prior, n_classes,
                   class_names, mc_forward, plot=True):
    print("\n" + "═"*72)
    print("  SIMULATION MODE 1 — STANDARD SCENARIO SWEEP")
    print("  Runs all 5 scenarios with synthetic stochastic input + MC Dropout")
    print("═"*72)

    print("\n  Showing Markov matrices...")
    plot_markov_matrices(markov_matrix, transition_only_markov, escalation_prior, class_names)

    all_results = {}
    for name, cfg in SCENARIOS.items():
        results = run_scenario(
            name, cfg['sequence'], cfg['ctx'],
            cal_xgb, lstm_model, le, pool, noise_scale,
            markov_matrix, transition_only_markov, escalation_prior,
            n_classes, class_names, mc_forward,
        )
        all_results[name] = results

        print_scenario_header(name, cfg['sequence'], cfg['ctx'])
        print(f"  ℹ  {cfg['desc']}\n")

        for r in results:
            print_step(r, class_names, markov_matrix, show_lookahead=True)

        alert_steps = [r for r in results if r['alert_level'] in ('HIGH','MEDIUM')]
        if alert_steps:
            print_defense_actions(alert_steps[-1])

        print_scenario_summary(name, results, class_names)

        if plot:
            plot_scenario_dashboard(name, results, class_names, markov_matrix)

    return all_results


# ── Mode 2: Real Sample Mode ───────────────────────────────────────────────────
def demo_real_samples(cal_xgb, lstm_model, le, pool, noise_scale, markov_matrix,
                      transition_only_markov, escalation_prior, n_classes,
                      class_names, mc_forward, plot=True):
    """Uses actual samples from X_bal (training data)."""
    print("\n" + "═"*72)
    print("  SIMULATION MODE 2 — REAL SAMPLE INPUT")
    print("  Feature vectors are drawn from actual X_bal training samples")
    print("═"*72)

    cfg = SCENARIOS['B_SlowEscalation']
    results = run_scenario(
        'B_SlowEscalation (Real Samples)', cfg['sequence'], cfg['ctx'],
        cal_xgb, lstm_model, le, pool, noise_scale,
        markov_matrix, transition_only_markov, escalation_prior,
        n_classes, class_names, mc_forward,
        use_real_samples=True,
    )

    print_scenario_header('B_SlowEscalation (Real Samples)', cfg['sequence'], cfg['ctx'])
    for r in results:
        print_step(r, class_names, markov_matrix, show_lookahead=True)

    alert_steps = [r for r in results if r['alert_level'] in ('HIGH','MEDIUM')]
    if alert_steps:
        print_defense_actions(alert_steps[-1])
    print_scenario_summary('B_SlowEscalation (Real)', results, class_names)

    if plot:
        plot_scenario_dashboard('B_SlowEscalation_RealSamples', results, class_names, markov_matrix)


# ── Mode 3: Live Interactive Input ────────────────────────────────────────────
def demo_interactive(cal_xgb, lstm_model, le, pool, noise_scale, markov_matrix,
                     transition_only_markov, escalation_prior, n_classes,
                     class_names, mc_forward, **_):
    """Reviewer types a traffic class one-by-one; system forecasts in real time."""
    print("\n" + "═"*72)
    print("  SIMULATION MODE 3 — LIVE INTERACTIVE INPUT")
    print("  Enter current traffic class; system forecasts NEXT class in real time.")
    print("═"*72)
    print(f"  Valid inputs: {list(class_names)}")
    print("  Type 'done' to finish.\n")

    ctx_input = {}
    print("  Set context (press Enter for defaults):")
    try:
        hour       = input("    Hour of day (0-23) [14]: ").strip()
        device     = input("    Device type (workstation/server/camera/sensor) [workstation]: ").strip()
        country    = input("    Country code [US]: ").strip()
        pkts       = input("    Packets per second [300]: ").strip()
        unique_d   = input("    Unique destinations [10]: ").strip()
        repeat_c   = input("    Threat repeat count [0]: ").strip()
        ctx_input  = dict(
            hour         = int(hour)     if hour     else 14,
            device_type  = device        if device   else 'workstation',
            country_code = country       if country  else 'US',
            pkts_per_sec = float(pkts)   if pkts     else 300,
            unique_dsts  = int(unique_d) if unique_d else 10,
            repeat_count = int(repeat_c) if repeat_c else 0,
        )
    except (EOFError, KeyboardInterrupt):
        ctx_input = dict(hour=14, device_type='workstation', country_code='US',
                         pkts_per_sec=300, unique_dsts=10, repeat_count=0)
        print("  (Using defaults)")

    ctx_score, _ = compute_context_score(**ctx_input)
    print(f"\n  ✅ Context loaded → score={ctx_score:.3f}\n")

    xgb_history   = deque(maxlen=WINDOW)
    lstm_history  = deque(maxlen=WINDOW)
    state_history = []
    step          = 0
    class_list    = [str(c) for c in class_names]

    print(f"  {'t':<4} {'Current':<16} {'→ Forecast':<16} {'Risk':<7} {'Alert':<10} Uncertainty")
    print(f"  {'─'*3} {'─'*15} {'─'*15} {'─'*6} {'─'*9} {'─'*10}")

    while True:
        try:
            raw = input(f"\n  Enter traffic class at t={step} > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ('done', 'quit', 'exit', 'q'):
            break

        matched = next((c for c in class_list if raw.lower() in c.lower() or c.lower() in raw.lower()), None)
        if matched is None:
            print(f"  ⚠  Unknown: '{raw}'. Valid: {class_list}")
            continue

        cls_name = matched
        state_history.append(cls_name)

        xgb_proba = simulate_window(cls_name, step, pool, noise_scale, cal_xgb)
        xgb_history.append(xgb_proba)
        lstm_history.append(xgb_proba)

        if len(lstm_history) >= WINDOW:
            lstm_inp               = np.array(list(lstm_history)).reshape(1, WINDOW, n_classes)
            lstm_proba, uncertainty, _ = lstm_predict_mc(lstm_inp, mc_forward)
        else:
            lstm_proba  = np.ones(n_classes) / n_classes
            uncertainty = 0.0

        csi    = int(le.transform([cls_name])[0])
        result = decision_engine_v3(
            xgb_proba, lstm_proba, csi, step,
            markov_matrix, transition_only_markov, escalation_prior,
            n_classes, class_names,
            context_kwargs=ctx_input, uncertainty=uncertainty,
            state_history=state_history, xgb_history=xgb_history,
        )

        alert_str = f"{ALERT_ICON[result['alert_level']]} {result['alert_level']}"
        pred_icon = CLASS_ICON.get(result['predicted_next_class'], '')
        print(f"  t{step:<3} {CLASS_ICON.get(cls_name,'')}{cls_name:<15} "
              f"→ {pred_icon}{result['predicted_next_class']:<15} "
              f"{result['final_risk']:.3f}  {alert_str:<16} "
              f"{result['uncertainty']:.3f} ({result['uncertainty_flag']})")

        if result['alert_level'] in ('HIGH', 'MEDIUM'):
            print(f"  ⚠  ACTIONS → {', '.join(result['actions'])}")

        forecasts = markov_lookahead(result['predicted_next_class'],
                                     class_names, markov_matrix, n_steps=3)
        top_fc = [f"{max(d, key=d.get)}@t+{i+1}({max(d.values()):.2f})" for i, d in enumerate(forecasts)]
        print(f"  🔮 Markov lookahead: {' | '.join(top_fc)}")

        step += 1


# ── Mode 4: Stress Test ────────────────────────────────────────────────────────
def demo_stress_test(cal_xgb, lstm_model, le, pool, noise_scale, markov_matrix,
                     transition_only_markov, escalation_prior, n_classes,
                     class_names, mc_forward, **_):
    print("\n" + "═"*72)
    print("  SIMULATION MODE 4 — FULL STRESS TEST + SUMMARY TABLE")
    print("  Runs all 5 scenarios × 3 seeds → average forecast accuracy")
    print("═"*72)

    rows = []
    for name, cfg in SCENARIOS.items():
        seed_accs = []
        for seed in [0, 1, 2]:
            np.random.seed(seed)
            results = run_scenario(
                name, cfg['sequence'], cfg['ctx'],
                cal_xgb, lstm_model, le, pool, noise_scale,
                markov_matrix, transition_only_markov, escalation_prior,
                n_classes, class_names, mc_forward,
            )
            valid = [r for r in results if r['true_next_class'] != '(FORECAST)']
            n_c   = sum(1 for r in valid if r['predicted_next_class'] == r['true_next_class'])
            seed_accs.append(n_c / len(valid) * 100 if valid else 0)

        np.random.seed(42)
        results  = run_scenario(
            name, cfg['sequence'], cfg['ctx'],
            cal_xgb, lstm_model, le, pool, noise_scale,
            markov_matrix, transition_only_markov, escalation_prior,
            n_classes, class_names, mc_forward,
        )
        valid    = [r for r in results if r['true_next_class'] != '(FORECAST)']
        n_c      = sum(1 for r in valid if r['predicted_next_class'] == r['true_next_class'])
        n_high   = sum(1 for r in results if r['alert_level'] == 'HIGH')
        avg_risk = np.mean([r['final_risk']             for r in results])
        avg_unc  = np.mean([r['uncertainty']            for r in results])
        avg_tl   = np.mean([r['transition_likelihood']  for r in results])

        rows.append({
            'Scenario'    : name,
            'Avg Acc %'   : f"{np.mean(seed_accs):.0f}±{np.std(seed_accs):.0f}",
            'HIGH Alerts' : n_high,
            'Avg Risk'    : f"{avg_risk:.3f}",
            'Avg Unc'     : f"{avg_unc:.3f}",
            'Avg TL'      : f"{avg_tl:.3f}",
        })
        print(f"  ✅ {name:<24} Acc={np.mean(seed_accs):.0f}%  "
              f"HIGH={n_high}  Risk={avg_risk:.3f}  Unc={avg_unc:.3f}")

    print(f"\n  {'─'*70}")
    print(f"  {'SCENARIO':<24} {'Acc%':>10} {'HIGH':>6} {'AvgRisk':>9} {'AvgUnc':>8} {'AvgTL':>7}")
    print(f"  {'─'*24} {'─'*10} {'─'*6} {'─'*9} {'─'*8} {'─'*7}")
    for r in rows:
        print(f"  {r['Scenario']:<24} {r['Avg Acc %']:>10} {r['HIGH Alerts']:>6} "
              f"{r['Avg Risk']:>9} {r['Avg Unc']:>8} {r['Avg TL']:>7}")
    print(f"  {'─'*70}")

    print("""
  ─────────────────────────────────────────────────────────────────────────
  PIPELINE SUMMARY
  ─────────────────────────────────────────────────────────────────────────
  XGBoost  ( 5%) → "What is happening RIGHT NOW?"  instant tabular detection
  LSTM     (15%) → "What PATTERN is forming?"       temporal sequence learning
                   ★ MC Dropout: 30 passes → mean proba + entropy
  Adaptive (80%) → "What is LIKELY NEXT?"
  Markov           = stability×raw + context×(transition + escalation prior)
  Context        → "How SUSPICIOUS is this?"
                   5D: time, device, network, threat_history, geolocation
  Uncertainty    → "How CONFIDENT is the LSTM?"
                   High entropy → downgrade HIGH to MEDIUM for human review
  ─────────────────────────────────────────────────────────────────────────
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN MENU
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═"*70 + "╗")
    print("║  🔒  CONTEXT-AWARE CYBER THREAT FORECASTING — FYP FINAL DEMO     ║")
    print("║      XGBoost + LSTM (MC Dropout) + Adaptive Markov v3            ║")
    print("║      Next-State Forecasting + Context Engine + Proactive Defense ║")
    print("╚" + "═"*70 + "╝")

    # ── Load everything ───────────────────────────────────────────────────────
    cal_xgb, lstm_model, le, X_bal, y_bal, class_names, features = load_artifacts()
    n_classes = len(class_names)

    print("  Rebuilding stochastic modules from saved data...")
    markov_matrix          = build_markov_matrix(y_bal.astype(int), n_classes)
    transition_only_markov = build_transition_only_markov(markov_matrix)
    escalation_prior       = build_escalation_prior(class_names)
    pool, noise_scale      = build_sample_pool(X_bal, y_bal, class_names, le)
    mc_forward             = setup_mc_dropout(lstm_model)

    print("  ✅ Markov matrix (raw)     ready")
    print("  ✅ Transition-only Markov  ready")
    print("  ✅ Escalation prior        ready  (domain kill-chain knowledge)")
    print("  ✅ Sample pool + noise     ready  (50 samples/class, 5% noise)")
    print("  ✅ MC Dropout              ready  (30 passes per inference)")
    print()

    # Print Markov matrices as DataFrames
    print("  Raw Markov Transition Matrix:")
    print(pd.DataFrame(np.round(markov_matrix, 3),
                       index=class_names, columns=class_names).to_string())
    print("\n  Escalation Prior (domain knowledge):")
    print(pd.DataFrame(np.round(escalation_prior, 3),
                       index=class_names, columns=class_names).to_string())
    print()

    # ── Menu ──────────────────────────────────────────────────────────────────
    MODES = {
        '1': ('Standard Scenario Sweep (5 scenarios, synthetic)',    demo_scenarios),
        '2': ('Real Sample Input (actual X_bal feature vectors)',     demo_real_samples),
        '3': ('Live Interactive — type class step by step',          demo_interactive),
        '4': ('Full Stress Test × 3 seeds + Summary Table',          demo_stress_test),
        '5': ('Run ALL modes in sequence',                           None),
    }

    print("  SELECT SIMULATION MODE:")
    for k, (desc, _) in MODES.items():
        print(f"    [{k}] {desc}")
    print()

    try:
        choice = input("  Enter choice [1-5, default=1]: ").strip() or '1'
    except (EOFError, KeyboardInterrupt):
        choice = '1'

    shared_args = (cal_xgb, lstm_model, le, pool, noise_scale,
                   markov_matrix, transition_only_markov, escalation_prior,
                   n_classes, class_names, mc_forward)

    if choice == '5':
        demo_scenarios(*shared_args)
        demo_real_samples(*shared_args)
        demo_stress_test(*shared_args)
    elif choice in MODES and MODES[choice][1] is not None:
        MODES[choice][1](*shared_args)
    else:
        print("  Invalid choice — running Mode 1 (Standard Scenario Sweep)")
        demo_scenarios(*shared_args)

    print("\n  ✅ Demo complete.\n")


if __name__ == '__main__':
    main()
