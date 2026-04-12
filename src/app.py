import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

"""
All model weights, scaling parameters, test predictions, and career data are
derived by running the actual Gibbs sampler on the training CSVs at startup.
"""

# Resolve data directory relative to this file.
# app.py lives in  src/  and the CSVs live in  src/data/
DATA_DIR = Path(__file__).parent / "data"

def dp(filename: str) -> str:
    """Return absolute path to a data file."""
    return str(DATA_DIR / filename)

st.set_page_config(page_title="Football Transfer Value Forecasting", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .hero {
    background: #0a0a0a; border-radius: 4px;
    padding: 2.4rem 2.5rem 1.9rem; color: #f0f0f0;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
  }
  .hero::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,#00c896,#00a8ff,#00c896);
  }
  .hero h1 { font-size:1.85rem; font-weight:700; letter-spacing:-0.02em; margin:0 0 .4rem; color:#fff; }
  .hero .sub { font-size:0.88rem; color:#888; font-family:'IBM Plex Mono',monospace; margin:0; }

  .slabel {
    font-family:'IBM Plex Mono',monospace; font-size:0.70rem; color:#888;
    letter-spacing:0.10em; text-transform:uppercase; margin-bottom:0.5rem;
  }
  .step {
    background:#f7f7f7; border-left:3px solid #00c896;
    border-radius:0 6px 6px 0; padding:0.75rem 1.1rem; margin-bottom:0.55rem;
  }
  .step .snum { font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
    color:#00c896; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; margin-bottom:0.15rem; }
  .step .stitle { font-weight:700; font-size:0.92rem; color:#111; margin-bottom:0.15rem; }
  .step .sbody  { font-size:0.83rem; color:#555; line-height:1.5; }

  .callout {
    background:#f0faf6; border-left:3px solid #00c896;
    border-radius:0 6px 6px 0; padding:0.75rem 1.1rem; margin:0.8rem 0;
    font-size:0.87rem; color:#333; line-height:1.5;
  }
</style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Model Evaluation", "Value Estimator"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:#888'>"
    "Lance Hendricks and Maxim Izosimov<br>Northeastern University, April 2026</span>",
    unsafe_allow_html=True
)

# Data loading and model fitting

LEAGUE_LEVELS = ['Bundesliga', 'Serie_A', 'Ligue_1', 'La_Liga', 'EPL']
LEAGUE_MAP_M  = {'ES1':'La_Liga','FR1':'Ligue_1','GR1':'Bundesliga','GB1':'EPL','IT1':'Serie_A'}

def winsorize(train, test, cols):
    caps = {}
    for col in cols:
        cap = train[col].quantile(0.99)
        train[col] = train[col].clip(upper=cap)
        test[col]  = test[col].clip(upper=cap)
        caps[col] = cap
    return train, test, caps

def one_hot(df):
    for lev in LEAGUE_LEVELS[1:]:
        df[lev] = (df['league'] == lev).astype(float)
    return df

def minmax_scale(train, test, continuous_cols):
    col_mins = train[continuous_cols].min()
    col_maxs = train[continuous_cols].max()
    for col in continuous_cols:
        train[col] = (train[col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
        test[col]  = (test[col]  - col_mins[col]) / (col_maxs[col] - col_mins[col])
    return train, test, col_mins, col_maxs

def run_gibbs(X_train, y_train, prior_w, n_total=1220, burn=220, seed=42):
    p     = len(prior_w)
    pSinv = np.eye(p)
    pa, pb = 1.0, 1.0
    np.random.seed(seed)
    wt, s2 = np.zeros(p), 1.0
    XtX, Xty = X_train.T @ X_train, X_train.T @ y_train
    n = len(y_train)
    all_w, all_s = np.zeros((n_total, p)), np.zeros(n_total)
    for i in range(n_total):
        V  = np.linalg.inv((1/s2)*XtX + pSinv)
        M  = V @ ((1/s2)*Xty + pSinv @ prior_w)
        wt = np.random.multivariate_normal(M, V)
        r  = y_train - X_train @ wt
        A  = (2*pa + n) / 2
        B  = (2*pb + r @ r) / 2
        s2 = 1.0 / np.random.gamma(A, 1.0/B)
        all_w[i], all_s[i] = wt, s2
    return all_w[burn:], all_s[burn:]


@st.cache_data(show_spinner="Fitting models on real training data...")
def load_and_fit():
    # Forwards
    f_train = pd.read_csv(dp('F_stats_values_train.csv'))
    f_test  = pd.read_csv(dp('F_stats_values_test.csv'))

    perf_cols = ['xG_per_90', 'xA_per_90', 'xGChain_per_90']
    f_train, f_test, winsor_caps_f = winsorize(f_train, f_test, perf_cols)

    f_train['age_log'] = np.log(f_train['age'])
    f_test['age_log']  = np.log(f_test['age'])

    f_train = one_hot(f_train)
    f_test  = one_hot(f_test)

    continuous_cols = ['xG_per_90', 'xA_per_90', 'xGChain_per_90', 'age_log', 'year']
    f_train, f_test, col_mins_f, col_maxs_f = minmax_scale(f_train, f_test, continuous_cols)

    # Forward feature order matches forward_bayesian_model.R:
    # xG, xA, xGChain, age_log, year, SerieA, Ligue1, LaLiga, EPL
    feat_f = ['xG_per_90','xA_per_90','xGChain_per_90','age_log','year',
              'Serie_A','Ligue_1','La_Liga','EPL']
    X_train_f = np.hstack([f_train[feat_f].values, np.ones((len(f_train), 1))])
    y_train_f = np.log(f_train['value'].values)
    X_test_f  = np.hstack([f_test[feat_f].values,  np.ones((len(f_test),  1))])
    y_test_f  = np.log(f_test['value'].values)

    # Prior from forward_bayesian_model.R
    prior_w_f = np.array([0.5, 0.5, 1.0, -3.0, 0.3, 0.0, -0.2, 0.7, 1.0, np.log(1e6)])
    samples_w_f, samples_s_f = run_gibbs(X_train_f, y_train_f, prior_w_f)

    # Midfielders
    m_all = pd.read_csv(dp('M_stats_values.csv'))
    m_all['league'] = m_all['league'].map(LEAGUE_MAP_M)

    np.random.seed(42)
    all_players = m_all['player_id'].unique()
    train_players = np.random.choice(all_players, size=int(0.8*len(all_players)), replace=False)
    m_train = m_all[m_all['player_id'].isin(train_players)].copy()
    m_test  = m_all[~m_all['player_id'].isin(train_players)].copy()

    m_train, m_test, winsor_caps_m = winsorize(m_train, m_test, perf_cols)
    m_train['age_log'] = np.log(m_train['age'])
    m_test['age_log']  = np.log(m_test['age'])
    m_train = one_hot(m_train)
    m_test  = one_hot(m_test)

    m_train, m_test, col_mins_m, col_maxs_m = minmax_scale(m_train, m_test, continuous_cols)

    # Midfielder feature order matches midfielder_bayesian_model.R:
    # xG, xA, xGChain, year, age_log, SerieA, Ligue1, EPL, LaLiga
    feat_m = ['xG_per_90','xA_per_90','xGChain_per_90','year','age_log',
              'Serie_A','Ligue_1','EPL','La_Liga']
    X_train_m = np.hstack([m_train[feat_m].values, np.ones((len(m_train), 1))])
    y_train_m = np.log(m_train['value'].values)
    X_test_m  = np.hstack([m_test[feat_m].values,  np.ones((len(m_test),  1))])
    y_test_m  = np.log(m_test['value'].values)

    # Prior from midfielder_bayesian_model.R
    prior_w_m = np.array([0.2, 0.2, 2.0, 0.3, -3.0, 0.0, -0.2, 1.0, 0.7, np.log(1e6)])
    samples_w_m, samples_s_m = run_gibbs(X_train_m, y_train_m, prior_w_m)

    # Test predictions for scatter plot (Forwards) 
    post_w_f   = samples_w_f.mean(axis=0)
    y_pred_f   = X_test_f @ post_w_f
    r2_f = 1 - np.sum((y_test_f-y_pred_f)**2) / np.sum((y_test_f-np.mean(y_test_f))**2)
    scatter_f  = pd.DataFrame({'actual': np.exp(y_test_f)/1e6, 'predicted': np.exp(y_pred_f)/1e6})

    # Test predictions for scatter plot (Midfielders) 
    post_w_m   = samples_w_m.mean(axis=0)
    y_pred_m   = X_test_m @ post_w_m
    r2_m = 1 - np.sum((y_test_m-y_pred_m)**2) / np.sum((y_test_m-np.mean(y_test_m))**2)
    scatter_m  = pd.DataFrame({'actual': np.exp(y_test_m)/1e6, 'predicted': np.exp(y_pred_m)/1e6})

    # Career arc: Marcus Rashford (Forward, test set, 32 blocks) 
    f_test_raw = pd.read_csv(dp('F_stats_values_test.csv'))
    for col in perf_cols:
        f_test_raw[col] = f_test_raw[col].clip(upper=winsor_caps_f[col])
    f_test_raw['age_log'] = np.log(f_test_raw['age'])
    f_test_raw = one_hot(f_test_raw)
    for col in continuous_cols:
        f_test_raw[col] = (f_test_raw[col] - col_mins_f[col]) / (col_maxs_f[col] - col_mins_f[col])
    p_df = f_test_raw[f_test_raw['player_name'] == 'Marcus Rashford'].copy()
    p_df['date'] = pd.to_datetime(p_df['date'])
    p_df = p_df.sort_values('date')
    X_p = np.hstack([p_df[feat_f].values, np.ones((len(p_df), 1))])
    np.random.seed(0)
    career_means, career_lowers, career_uppers, career_actuals = [], [], [], []
    for i in range(len(p_df)):
        draws = np.array([
            np.random.normal(X_p[i] @ samples_w_f[j], np.sqrt(samples_s_f[j]))
            for j in range(len(samples_w_f))
        ])
        career_means.append(np.exp(np.mean(draws)) / 1e6)
        career_lowers.append(np.exp(np.quantile(draws, 0.05)) / 1e6)
        career_uppers.append(np.exp(np.quantile(draws, 0.95)) / 1e6)
        career_actuals.append(p_df['value'].values[i] / 1e6)
    career_df = pd.DataFrame({
        'label': [f"{r['date'].strftime('%Y-%m')} (age {r['age']:.0f})"
                  for _, r in p_df.iterrows()],
        'actual':  career_actuals,
        'mean':    career_means,
        'lower':   career_lowers,
        'upper':   career_uppers,
    })

    return {
        'samples_w_f': samples_w_f, 'samples_s_f': samples_s_f,
        'samples_w_m': samples_w_m, 'samples_s_m': samples_s_m,
        'prior_w_f': prior_w_f,     'prior_w_m': prior_w_m,
        'col_mins_f': col_mins_f,   'col_maxs_f': col_maxs_f,
        'col_mins_m': col_mins_m,   'col_maxs_m': col_maxs_m,
        'winsor_caps_f': winsor_caps_f, 'winsor_caps_m': winsor_caps_m,
        'feat_f': feat_f, 'feat_m': feat_m,
        'scatter_f': scatter_f,     'scatter_m': scatter_m,
        'r2_f': r2_f,               'r2_m': r2_m,
        'career_df': career_df,
    }

data = load_and_fit()

# Helper: build feature row from user inputs
def build_x_row(position, xg, xa, xgchain, age, year, league):
    if position == "Forward":
        caps    = data['winsor_caps_f']
        col_min = data['col_mins_f']
        col_max = data['col_maxs_f']
    else:
        caps    = data['winsor_caps_m']
        col_min = data['col_mins_m']
        col_max = data['col_maxs_m']

    def sc(v, col):
        v = min(v, caps[col])
        return np.clip((v - col_min[col]) / (col_max[col] - col_min[col] + 1e-12), 0, 1)

    age_log_val = np.log(max(age, 1))
    age_log_sc  = np.clip(
        (age_log_val - col_min['age_log']) / (col_max['age_log'] - col_min['age_log'] + 1e-12), 0, 1)
    year_sc     = np.clip(
        (year - col_min['year']) / (col_max['year'] - col_min['year'] + 1e-12), 0, 1)

    xg_s  = sc(xg,      'xG_per_90')
    xa_s  = sc(xa,      'xA_per_90')
    xgc_s = sc(xgchain, 'xGChain_per_90')

    league_oh_f = {"EPL":[0,0,0,1],"La Liga":[0,0,1,0],"Bundesliga":[0,0,0,0],
                   "Serie A":[1,0,0,0],"Ligue 1":[0,1,0,0]}
    league_oh_m = {"EPL":[0,0,1,0],"La Liga":[0,0,0,1],"Bundesliga":[0,0,0,0],
                   "Serie A":[1,0,0,0],"Ligue 1":[0,1,0,0]}

    if position == "Forward":
        # order: xG, xA, xGChain, age_log, year, SerieA, Ligue1, LaLiga, EPL, bias
        loh = league_oh_f[league]
        return np.array([xg_s, xa_s, xgc_s, age_log_sc, year_sc] + loh + [1.0])
    else:
        # order: xG, xA, xGChain, year, age_log, SerieA, Ligue1, EPL, LaLiga, bias
        loh = league_oh_m[league]
        return np.array([xg_s, xa_s, xgc_s, year_sc, age_log_sc] + loh + [1.0])

def posterior_draws(x_row, samples_w, samples_s, seed=42):
    np.random.seed(seed)
    return np.array([
        np.random.normal(x_row @ samples_w[j], np.sqrt(samples_s[j]))
        for j in range(len(samples_w))
    ])

PLOT_CFG = dict(plot_bgcolor="#fff", paper_bgcolor="#fff",
                font=dict(family="IBM Plex Sans", size=12))


# Page 1 project overview
if page == "Project Overview":
    st.markdown("""
    <div class="hero">
      <h1>Football Transfer Value Forecasting with Uncertainty</h1>
      <p class="sub">LSTM performance forecasting + Bayesian linear regression (Forwards and Midfielders) 2014 to 2025</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([5, 4], gap="large")

    with col_l:
        st.markdown('<div class="slabel">The Problem</div>', unsafe_allow_html=True)
        st.markdown("""
Football clubs spend billions annually on player transfers, yet valuation largely comes down to
scouting opinions and market speculation. Clubs routinely overpay for players whose performance
does not justify the price and overlook undervalued talent entirely.

Two questions sit at the core of this problem:
1. **Given how a player is currently performing, what are they worth today?**
2. **Given their performance history, how will they perform and what will that be worth in the future?**

This project builds a modular two-part framework to answer both, and crucially, to quantify
**how uncertain those answers are**. No comparable work in the literature provides uncertainty
estimates, every existing model returns only a point prediction.
        """)
        st.markdown("---")
        st.markdown('<div class="slabel">Data and Processing Pipeline</div>', unsafe_allow_html=True)

        for num, title, body in [
            ("Step 1", "Raw match data via Understat API",
             "Per-match statistics collected across Europe's top-5 leagues (EPL, La Liga, Bundesliga, "
             "Serie A, Ligue 1) from 2014 onward, via the Understat API. Covers forwards and midfielders "
             "only, Understat does not provide meaningful defensive statistics."),
            ("Step 2", "Aggregate into 10-game blocks",
             "Rather than modelling noisy per-game figures, consecutive matches are grouped into "
             "rolling blocks of 10 games, roughly one quarter of a season. This smooths out "
             "match-to-match variance while preserving granular form trends across a career."),
            ("Step 3", "Normalise to per-90 minutes",
             "All statistics are expressed per 90 minutes of playing time so that blocks are directly "
             "comparable regardless of how many minutes were accumulated. This matters especially "
             "for players who appear frequently as substitutes."),
            ("Step 4", "Feature selection: drop multicollinear statistics",
             "Of the five Understat metrics initially collected (xG, xA, xGChain, xGBuildup, key passes), "
             "xGBuildup and key passes were dropped due to significant multicollinearity with the "
             "retained features. Only Expected Goals (xG), Expected Assists (xA), and xGChain per 90 "
             "are used as performance inputs."),
            ("Step 5", "As-of merge with Transfermarkt valuations",
             "For each Transfermarkt valuation date, the most recent completed 10-game block prior to "
             "that date is matched in. Player age (log-transformed), calendar year (to capture "
             "transfer market inflation), and domestic league are appended as additional features."),
            ("Step 6", "80/20 player-based train and test split",
             "Players, not individual observations, are randomly assigned to train or test sets. "
             "This prevents the same player from appearing in both splits, ensuring a fair "
             "evaluation on entirely unseen players."),
        ]:
            st.markdown(f"""<div class="step">
              <div class="snum">{num}</div>
              <div class="stitle">{title}</div>
              <div class="sbody">{body}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="slabel">Performance Features</div>', unsafe_allow_html=True)
        st.markdown("""
All three retained statistics are measured **per 90 minutes** across a 10-game block:

**Expected Goals (xG) per 90** is the sum of shot-scoring probabilities in the block, where each
shot is assigned a probability based on location, angle, and assist type. xG captures goal-scoring
threat more reliably than actual goals, which are subject to luck. Typical range: 0.05 (fringe player)
to 0.99 (top forwards, after 99th-percentile winsorizing applied during training).

**Expected Assists (xA) per 90** is the expected-goals value of passes that immediately preceded a
shot. xA measures creative contribution, specifically how reliably a player puts teammates into goal-scoring
positions. Typical range: 0.05 to 0.51.

**xGChain per 90** is the total xG of every possession the player was involved in at any point
that ended in a shot. Unlike xG and xA, xGChain credits players anywhere in the build-up, not
just on the final pass or shot. It is especially diagnostic for midfielders whose value comes
from linking play. Typical range: 0.25 to 1.21 (midfielders) or 1.34 (forwards).
        """)

        st.markdown("---")
        st.markdown('<div class="slabel">Two-Part Framework</div>', unsafe_allow_html=True)
        st.markdown("""
**Part 1: LSTM Performance Forecaster**

A Long Short-Term Memory (LSTM) neural network takes the last 10 game blocks as input (each a
3-vector of per-90 xG, xA, xGChain) and predicts the next block. By chaining predictions
autoregressively, the model forecasts performance multiple blocks into the future. Trained
separately for forwards and midfielders via 80/20 player-based split. Hyperparameters tuned by
grid search, best config: learning rate 0.001, 1 hidden layer, width 64, 20 epochs.

**Part 2: Bayesian Linear Regression**

Maps performance features (real or LSTM-generated) plus log(age), year, and league one-hot
encodings to log(market value). Fitted via Gibbs sampling with a Gaussian likelihood, Gaussian
weight priors, and inverse-gamma variance prior. The posterior is sampled using 18% burn-in
and 1,000 kept samples. Every prediction is a full **posterior predictive distribution**, not
just a point estimate.
        """)

        st.markdown("---")
        st.markdown('<div class="slabel">Why Separate Models per Position?</div>', unsafe_allow_html=True)
        st.markdown("""
The relationship between statistics and market value differs fundamentally by position.
For a **forward**, Expected Goals (xG) is the dominant signal: clubs primarily pay for
goal-scoring threat. For a **midfielder**, xGChain carries far more weight because their value
lies in build-up involvement and linking play, not finishing.

This is encoded directly in the prior weights below. A single pooled model cannot
simultaneously upweight xG for forwards and xGChain for midfielders, which is why the
two models are fit and evaluated separately.
        """)

        st.markdown('<div class="slabel" style="margin-top:1rem">Informative Prior Weights (from R code)</div>', unsafe_allow_html=True)
        st.markdown("""
The table below shows the prior mean for each feature weight. These are not guesses, they encode
specific domain beliefs about how each factor should influence log(market value). The Gibbs sampler
then updates these priors using the training data to produce posterior estimates.
A **prior of +2.0 on xGChain for midfielders** reflects the belief that build-up involvement
is the strongest per-unit predictor of their value. A **prior of -3.0 on age** (which is
log-transformed) reflects the steep premium clubs pay for younger players with longer resale
windows. The Premier League prior of +1.0 reflects that EPL clubs consistently pay more for
equivalent players relative to the Bundesliga baseline.
        """)

        prior_df = pd.DataFrame({
            "Feature": ["xG per 90", "xA per 90", "xGChain per 90",
                        "Age (log)", "Year", "Premier League",
                        "La Liga", "Ligue 1", "Serie A", "Intercept"],
            "MF prior": ["+0.2", "+0.2", "+2.0", "-3.0", "+0.3",
                         "+1.0", "+0.7", "-0.2", "0.0", "log(1M)"],
            "FW prior": ["+0.5", "+0.5", "+1.0", "-3.0", "+0.3",
                         "+1.0", "+0.7", "-0.2", "0.0", "log(1M)"],
            "Why": [
                "Forwards: goal threat is primary signal",
                "Midfielders: goals less central, xGChain carries more",
                "Midfielders: build-up involvement most diagnostic",
                "Strong negative: younger players command large premium",
                "Small positive: transfer fee inflation over time",
                "EPL clubs pay the highest fees globally",
                "La Liga next highest after Premier League",
                "Ligue 1 slight discount vs Bundesliga baseline",
                "Serie A roughly equivalent to Bundesliga",
                "Baseline log-value for young low-output player",
            ],
        })
        st.dataframe(prior_df, width='stretch', hide_index=True, height=388)


# Page 2: model evaluation
elif page == "Model Evaluation":
    st.markdown("## Model Evaluation")
    st.markdown("""
    Both plots below are produced from the **actual Gibbs sampler run on the real training data**,
    evaluated against the real held-out test set. Nothing here is simulated.

    The scatter plot shows all 4,153 forward player-valuation pairs from the test set: the
    posterior mean prediction vs. the Transfermarkt ground truth, on a log-log scale.
    The career plot shows the full posterior predictive distribution over Marcus Rashford's
    market value at every 10-game block in the test set, with the 90% credible interval ribbon.
    """)

    st.markdown("---")

    # Scatter plot
    st.markdown("### Predicted vs. Actual Market Value: Forward Test Set")
    st.markdown(
        "<span style='font-size:0.83rem;color:#555'>"
        "Each point is one player-valuation pair from the held-out test set (4,153 pairs across "
        f"all five leagues, 2014 to 2025). The red diagonal is perfect prediction. "
        f"Log scale on both axes because transfer fees span several orders of magnitude "
        f"(from under 1M euros to 200M euros). The model achieves R-squared of "
        f"<strong>{data['r2_f']:.2f}</strong> on log-scale predictions."
        "</span>", unsafe_allow_html=True
    )
    sc = data['scatter_f']
    # Clip extreme outliers for display (top 0.5%) so axis is readable
    clip_a = sc['actual'].quantile(0.995)
    clip_p = sc['predicted'].quantile(0.995)
    sc_plot = sc[(sc['actual'] <= clip_a) & (sc['predicted'] <= clip_p)]

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=sc_plot['actual'], y=sc_plot['predicted'], mode="markers",
        marker=dict(color="rgba(0,168,255,0.25)", size=4),
        name="Player-valuation pair"
    ))
    mn, mx = sc_plot['actual'].min(), sc_plot['actual'].max()
    fig_sc.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#e63946", width=1.8), name="Perfect prediction"
    ))
    fig_sc.update_layout(
        **PLOT_CFG,
        xaxis=dict(title="Actual Value (euros M, log scale)", type="log",
                   gridcolor="#eee", tickformat=".0f"),
        yaxis=dict(title="Predicted Value (euros M, log scale)", type="log",
                   gridcolor="#eee", tickformat=".0f"),
        height=440, margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_sc, width='stretch')

    st.markdown(
        '<div class="callout">'
        'The model correctly captures the broad shape of the transfer market: cheap players '
        'are predicted cheaply and expensive players are predicted expensively. Individual '
        'predictions carry substantial variance, which is expected. Transfer fees are influenced by '
        'factors well beyond on-pitch statistics, including injury history, contract length, and '
        'club financial situations. The wide credible intervals on the career plot below reflect '
        'this honestly, rather than hiding it behind a false point estimate.'
        '</div>', unsafe_allow_html=True
    )

    st.markdown("---")

    # Career arc: Rashford
    st.markdown("### Marcus Rashford: Predicted Value Over Career (Forward Test Set)")
    st.markdown(
        "<span style='font-size:0.83rem;color:#555'>"
        "Rashford appears 32 times in the forward test set, making him one of the most "
        "extensively covered players for evaluation. Each point represents one 10-game block. "
        "The blue ribbon is the <strong>90% posterior credible interval</strong>: the model "
        "assigns 90% probability to the true value lying within this band at that time. "
        "The wide ribbon is not a failure of the model, it is an honest reflection of how much "
        "uncertainty exists when predicting transfer fees from statistics alone."
        "</span>", unsafe_allow_html=True
    )

    cd = data['career_df']
    n  = len(cd)
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=list(range(n)) + list(range(n))[::-1],
        y=list(cd['upper']) + list(cd['lower'])[::-1],
        fill="toself", fillcolor="rgba(0,168,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="90% Credible Interval", hoverinfo="skip"
    ))
    fig_c.add_trace(go.Scatter(
        x=list(range(n)), y=cd['mean'],
        mode="lines+markers", name="Posterior Mean",
        line=dict(color="#00a8ff", width=2.2), marker=dict(size=5)
    ))
    fig_c.add_trace(go.Scatter(
        x=list(range(n)), y=cd['actual'],
        mode="lines+markers", name="Transfermarkt Value",
        line=dict(color="#e63946", width=2.2, dash="dot"),
        marker=dict(size=5, symbol="diamond")
    ))
    fig_c.update_layout(
        **PLOT_CFG,
        xaxis=dict(tickvals=list(range(n)), ticktext=cd['label'].tolist(),
                   tickangle=-40, title="Block (date and age)", gridcolor="#eee"),
        yaxis=dict(title="Market Value (euros millions)", gridcolor="#eee"),
        height=460, margin=dict(t=20, b=100),
        legend=dict(orientation="h", y=-0.38),
    )
    st.plotly_chart(fig_c, width='stretch')

    st.markdown(
        '<div class="callout">'
        'Unlike Aydemir et al. (gradient boosting), Tamim et al. (random forest), and van Arem '
        'et al. (tree-based temporal features), all of which return a single point estimate, '
        'this framework expresses every valuation as a probability distribution. A club can ask: '
        '"Given that the 90% credible interval at this point spans from X to Y, is the asking '
        'fee a reasonable risk?" No point-estimate model supports that question.'
        '</div>', unsafe_allow_html=True
    )

# Page 3: value estimator (interactive)
else:
    st.markdown("## Value Estimator")
    st.markdown("""
    Enter a player's statistics and context below. The Bayesian model, fit on real training data,
    computes the full posterior predictive distribution over their market value. The model and
    scaling parameters are identical to those used in the paper evaluation.

    All performance statistics are **per 90 minutes** across a 10-game block (approximately one
    quarter of a season). Inputs are winsorized and min-max scaled exactly as during training.
    """)

    st.markdown("---")
    st.markdown("### Player Inputs")

    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        position = st.radio("Position", ["Forward", "Midfielder"], horizontal=True)
    with ci2:
        league = st.selectbox("League", ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
    with ci3:
        age = st.slider("Age", 17, 38, 23)

    if position == "Forward":
        xg_max, xa_max, xgc_max = 0.99, 0.51, 1.34
        xg_def, xa_def, xgc_def = 0.28, 0.14, 0.50
    else:
        xg_max, xa_max, xgc_max = 0.67, 0.51, 1.21
        xg_def, xa_def, xgc_def = 0.12, 0.13, 0.43

    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        st.markdown("**Expected Goals (xG) / 90**")
        st.caption("Shot-scoring probability. Measures goal-scoring threat. "
                   f"Training cap (99th pct): {xg_max:.2f}")
        xg = st.slider("xG / 90", 0.0, xg_max, xg_def, 0.01, label_visibility="collapsed")
    with sp2:
        st.markdown("**Expected Assists (xA) / 90**")
        st.caption("xG value of key passes made. Measures creativity. "
                   f"Training cap (99th pct): {xa_max:.2f}")
        xa = st.slider("xA / 90", 0.0, xa_max, xa_def, 0.01, label_visibility="collapsed")
    with sp3:
        st.markdown("**xGChain / 90**")
        st.caption("xG of all possessions involving the player ending in a shot. "
                   f"Measures build-up involvement. Training cap: {xgc_max:.2f}")
        xgchain = st.slider("xGChain / 90", 0.0, xgc_max, xgc_def, 0.01, label_visibility="collapsed")
    with sp4:
        st.markdown("**Season Year**")
        st.caption("Captures transfer market inflation. Training data spans 2014 to 2026.")
        year = st.slider("Year", 2015, 2025, 2023, label_visibility="collapsed")

    samples_w = data['samples_w_f'] if position == "Forward" else data['samples_w_m']
    samples_s = data['samples_s_f'] if position == "Forward" else data['samples_s_m']

    x_row     = build_x_row(position, xg, xa, xgchain, age, year, league)
    log_draws = posterior_draws(x_row, samples_w, samples_s)
    val_draws = np.exp(log_draws)

    mean_val   = np.exp(np.mean(log_draws))
    median_val = np.exp(np.median(log_draws))
    lo_val     = np.exp(np.quantile(log_draws, 0.05))
    hi_val     = np.exp(np.quantile(log_draws, 0.95))
    ci_width   = hi_val - lo_val

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posterior Mean",   f"EUR {mean_val/1e6:.1f}M")
    m2.metric("Posterior Median", f"EUR {median_val/1e6:.1f}M")
    m3.metric("90% CI Lower",     f"EUR {lo_val/1e6:.1f}M")
    m4.metric("90% CI Upper",     f"EUR {hi_val/1e6:.1f}M")

    st.markdown("---")

    # Posterior distribution
    st.markdown("### Posterior Predictive Distribution")
    st.markdown(
        "<span style='font-size:0.85rem;color:#555'>"
        "Each bar shows how many of the 1,000 Gibbs posterior samples produced that market value "
        "when drawn from the likelihood. The green region is the 90% credible interval: the model "
        "assigns 90% probability to the true value falling within this band. "
        "The right-skewed shape reflects the log-normal structure of transfer fees: most players "
        "are worth modest amounts, with a long tail of very high valuations. The gap between the "
        "mean and median quantifies this skew."
        "</span>", unsafe_allow_html=True
    )

    clip_val = np.percentile(val_draws, 99)
    bins     = np.linspace(0, clip_val, 70)
    counts, edges = np.histogram(val_draws[val_draws <= clip_val], bins=bins)
    bw       = (edges[1] - edges[0]) / 1e6
    ci_mask  = (edges[:-1] >= lo_val) & (edges[1:] <= hi_val)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=edges[:-1]/1e6, y=counts, width=bw,
        marker_color="#cccccc", name="Outside 90% CI"
    ))
    fig_hist.add_trace(go.Bar(
        x=edges[:-1]/1e6, y=np.where(ci_mask, counts, 0), width=bw,
        marker_color="rgba(0,200,150,0.75)", name="90% Credible Interval"
    ))
    fig_hist.add_vline(x=mean_val/1e6, line_color="#0a0a0a", line_dash="dash", line_width=2,
                       annotation_text=f"Mean EUR {mean_val/1e6:.1f}M",
                       annotation_position="top right", annotation_font_size=11)
    fig_hist.add_vline(x=median_val/1e6, line_color="#e63946", line_dash="dot", line_width=2,
                       annotation_text=f"Median EUR {median_val/1e6:.1f}M",
                       annotation_position="top left", annotation_font_size=11)
    fig_hist.update_layout(
        **PLOT_CFG,
        xaxis_title="Market Value (euros millions)",
        yaxis_title="Number of posterior samples",
        barmode="overlay",
        height=360, margin=dict(t=20, b=50),
    )
    st.plotly_chart(fig_hist, width='stretch')

    # Age curve
    st.markdown("### Predicted Value Across Ages (All Other Inputs Fixed)")
    st.markdown(
        "<span style='font-size:0.85rem;color:#555'>"
        "Holding the performance statistics, league, and year fixed at the values above, "
        "this shows how the posterior mean prediction changes as the player ages. "
        "The age prior weight is -3.0 (applied to log-age after min-max scaling), reflecting "
        "the steep premium clubs pay for younger players with longer resale windows. "
        "The ribbon is the 90% credible interval, the red line marks the current age input."
        "</span>", unsafe_allow_html=True
    )

    ages_arr = np.arange(17, 39)
    c_means, c_los, c_his = [], [], []
    for a in ages_arr:
        row_a = build_x_row(position, xg, xa, xgchain, a, year, league)
        ld    = posterior_draws(row_a, samples_w, samples_s, seed=0)
        c_means.append(np.exp(np.mean(ld)) / 1e6)
        c_los.append(np.exp(np.quantile(ld, 0.05)) / 1e6)
        c_his.append(np.exp(np.quantile(ld, 0.95)) / 1e6)

    fig_age = go.Figure()
    fig_age.add_trace(go.Scatter(
        x=list(ages_arr) + list(ages_arr[::-1]),
        y=c_his + c_los[::-1],
        fill="toself", fillcolor="rgba(0,200,150,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip"
    ))
    fig_age.add_trace(go.Scatter(
        x=ages_arr, y=c_means, mode="lines+markers",
        name="Posterior Mean",
        line=dict(color="#00c896", width=2.3), marker=dict(size=5)
    ))
    fig_age.add_vline(x=age, line_color="#e63946", line_dash="dot", line_width=2,
                      annotation_text=f"Current age ({age})",
                      annotation_position="top right", annotation_font_size=11)
    fig_age.update_layout(
        **PLOT_CFG,
        xaxis_title="Age", yaxis_title="Market Value (euros millions)",
        height=340, margin=dict(t=20, b=50),
    )
    st.plotly_chart(fig_age, width='stretch')

    # Feature contributions table
    st.markdown("### Feature Contributions to This Valuation")
    st.markdown(
        "<span style='font-size:0.85rem;color:#555'>"
        "The posterior mean weight for each feature (averaged across all 1,000 Gibbs samples) "
        "multiplied by its scaled input value gives its additive contribution to log(market value). "
        "The largest absolute contributions are the primary drivers of this specific valuation. "
        "All continuous features have been min-max scaled to [0,1] using the training data ranges, "
        "and performance features were winsorized at the 99th percentile of training data before scaling."
        "</span>", unsafe_allow_html=True
    )

    post_w = samples_w.mean(axis=0)
    if position == "Forward":
        feat_names = ["Expected Goals (xG) per 90", "Expected Assists (xA) per 90",
                      "xGChain per 90", "Age (log, scaled)",
                      "Year (scaled)", "Serie A vs Bundesliga",
                      "Ligue 1 vs Bundesliga", "La Liga vs Bundesliga",
                      "Premier League vs Bundesliga", "Intercept"]
        feat_desc = [
            "Shot probability sum: primary goal-threat signal for forwards",
            "Key pass quality: creative contribution",
            "xG of all possessions player touched ending in shot",
            "Age premium: younger players command significantly higher fees",
            "Calendar year: transfer fees have inflated over time",
            "Serie A league adjustment (Bundesliga is reference)",
            "Ligue 1 league adjustment",
            "La Liga league adjustment",
            "Premier League adjustment: highest fees globally",
            "Baseline log-value when all inputs are zero",
        ]
    else:
        feat_names = ["Expected Goals (xG) per 90", "Expected Assists (xA) per 90",
                      "xGChain per 90", "Year (scaled)",
                      "Age (log, scaled)", "Serie A vs Bundesliga",
                      "Ligue 1 vs Bundesliga", "Premier League vs Bundesliga",
                      "La Liga vs Bundesliga", "Intercept"]
        feat_desc = [
            "Shot probability sum: less central for midfielders than for forwards",
            "Key pass quality: creative contribution",
            "xG of all possessions player touched ending in shot: most diagnostic for midfielders",
            "Calendar year: transfer fees have inflated over time",
            "Age premium: younger players command significantly higher fees",
            "Serie A league adjustment (Bundesliga is reference)",
            "Ligue 1 league adjustment",
            "Premier League adjustment: highest fees globally",
            "La Liga league adjustment",
            "Baseline log-value when all inputs are zero",
        ]

    contribs = post_w * x_row
    contrib_df = pd.DataFrame({
        "Feature":               feat_names,
        "What it measures":      feat_desc,
        "Scaled input":          [f"{v:.3f}" for v in x_row],
        "Posterior mean weight": [f"{v:+.3f}" for v in post_w],
        "Contribution (log EUR)":[f"{v:+.3f}" for v in contribs],
    })
    st.dataframe(contrib_df, width='stretch', hide_index=True, height=388)

    st.markdown(
        f'<div class="callout">'
        f'90% credible interval: EUR {lo_val/1e6:.1f}M to EUR {hi_val/1e6:.1f}M '
        f'(width EUR {ci_width/1e6:.1f}M). '
        f'Before committing to a transfer fee, a club can directly ask: given this range, '
        f'is paying EUR {mean_val/1e6:.0f}M (the posterior mean) a reasonable risk? '
        f'A narrow interval signals confidence, a wide one signals the estimate is speculative '
        f'and the investment carries more uncertainty than the headline number suggests.'
        f'</div>', unsafe_allow_html=True
    )