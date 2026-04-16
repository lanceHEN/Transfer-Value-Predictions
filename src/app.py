import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def dp(filename: str) -> str:
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

  .prefill-strip {
    background:#f0faf6; border-left:3px solid #00c896;
    border-radius:0 6px 6px 0; padding:0.75rem 1.1rem; margin-bottom:1rem;
  }
  .prefill-strip .ps-name { font-weight:700; font-size:0.97rem; color:#111; }
  .prefill-strip .ps-meta {
    font-size:0.76rem; color:#555; margin-top:0.1rem;
    font-family:'IBM Plex Mono',monospace;
  }

  /* Dark stat card */
  .stat-card {
    background: #1a1a1a;
    border-radius: 6px;
    padding: 0.85rem 1rem 0.7rem;
    margin-bottom: 0.5rem;
    min-height: 180px;
    display: flex;
    flex-direction: column;
  }
  .stat-card .sc-name {
    font-weight: 700;
    font-size: 0.92rem;
    color: #ffffff;
    display: block;
    margin-bottom: 0.3rem;
    flex-shrink: 0;
  }
  .stat-card .sc-desc {
    font-size: 0.78rem;
    color: #aaaaaa;
    line-height: 1.5;
    display: block;
    margin-bottom: 0.6rem;
    flex-grow: 1;
  }
  .stat-card .sc-benchmarks {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    flex-shrink: 0;
    margin-top: auto;
  }
  .stat-card .sc-bmark {
    background: #2e2e2e;
    border-radius: 4px;
    padding: 0.2rem 0.55rem;
    font-size: 0.70rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #cccccc;
    white-space: nowrap;
  }
  .stat-card .sc-bmark span {
    color: #00c896;
    font-weight: 700;
  }

  .val-summary {
    background:#f7f7f7; border-radius:6px; padding:1rem 1.2rem;
    margin-bottom:1.2rem; font-size:0.88rem; color:#333; line-height:1.7;
    border-left: 3px solid #00c896;
  }
  .val-summary strong { color:#111; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Value Estimator"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:#888'>"
    "Lance Hendricks and Maxim Izosimov<br>Northeastern University, April 2026</span>",
    unsafe_allow_html=True
)

# ── Constants ──────────────────────────────────────────────────────────────────
LEAGUE_LEVELS = ['Bundesliga', 'Serie_A', 'Ligue_1', 'La_Liga', 'EPL']
LEAGUE_MAP_M  = {'ES1':'La_Liga','FR1':'Ligue_1','GR1':'Bundesliga','GB1':'EPL','IT1':'Serie_A'}
LEAGUE_DISPLAY = {
    'EPL':'EPL', 'La_Liga':'La Liga', 'Bundesliga':'Bundesliga',
    'Serie_A':'Serie A', 'Ligue_1':'Ligue 1',
}
LEAGUES_UI = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# ── Preprocessing ──────────────────────────────────────────────────────────────
def winsorize(train, test, cols):
    caps = {}
    for col in cols:
        cap = train[col].quantile(0.99)
        train[col] = train[col].clip(upper=cap)
        test[col]  = test[col].clip(upper=cap)
        caps[col]  = cap
    return train, test, caps

def one_hot(df):
    for lev in LEAGUE_LEVELS[1:]:
        df[lev] = (df['league'] == lev).astype(float)
    return df

def minmax_scale(train, test, cols):
    col_mins = train[cols].min()
    col_maxs = train[cols].max()
    for col in cols:
        train[col] = (train[col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
        test[col]  = (test[col]  - col_mins[col]) / (col_maxs[col] - col_mins[col])
    return train, test, col_mins, col_maxs

# ── Gibbs sampler ──────────────────────────────────────────────────────────────
def run_gibbs(X_train, y_train, prior_w, n_total=6098, burn=1098, seed=42):
    p = len(prior_w)
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
        s2 = 1.0 / np.random.gamma((2*pa + n)/2, 1.0/((2*pb + r@r)/2))
        all_w[i], all_s[i] = wt, s2
    return all_w[burn:], all_s[burn:]


# ── Load, fit, build player index ─────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting models on training data...")
def load_and_fit():
    perf_cols = ['xG_per_90', 'xA_per_90', 'xGChain_per_90']
    cont_cols = ['xG_per_90', 'xA_per_90', 'xGChain_per_90', 'age_log', 'year']

    # Forwards
    f_tr = pd.read_csv(dp('F_stats_values_train.csv'))
    f_te = pd.read_csv(dp('F_stats_values_test.csv'))
    f_tr, f_te, wcaps_f = winsorize(f_tr, f_te, perf_cols)
    f_tr['age_log'] = np.log(f_tr['age']); f_te['age_log'] = np.log(f_te['age'])
    f_tr = one_hot(f_tr); f_te = one_hot(f_te)
    f_tr, f_te, cmin_f, cmax_f = minmax_scale(f_tr, f_te, cont_cols)
    feat_f = ['xG_per_90','xA_per_90','xGChain_per_90','age_log','year',
              'Serie_A','Ligue_1','La_Liga','EPL']
    Xf = np.hstack([f_tr[feat_f].values, np.ones((len(f_tr), 1))])
    yf = np.log(f_tr['value'].values)
    pw_f = np.array([1.5, 0.8, 0.4, -3.0, 0.3, 0.0, -0.2, 0.7, 1.0, np.log(1e6)])
    sw_f, ss_f = run_gibbs(Xf, yf, pw_f)
    f_thr = {k: float(np.percentile(f_tr['value'].values, p))
             for k, p in [('squad', 25), ('firstteam', 50), ('topflight', 75)]}

    # Midfielders
    m_tr = pd.read_csv(dp('M_stats_values_train.csv'))
    m_te = pd.read_csv(dp('M_stats_values_test.csv'))
    m_tr, m_te, wcaps_m = winsorize(m_tr, m_te, perf_cols)
    m_tr['age_log'] = np.log(m_tr['age']); m_te['age_log'] = np.log(m_te['age'])
    m_tr = one_hot(m_tr); m_te = one_hot(m_te)
    m_tr, m_te, cmin_m, cmax_m = minmax_scale(m_tr, m_te, cont_cols)
    feat_m = ['xG_per_90','xA_per_90','xGChain_per_90','year','age_log',
              'Serie_A','Ligue_1','EPL','La_Liga']
    Xm = np.hstack([m_tr[feat_m].values, np.ones((len(m_tr), 1))])
    ym = np.log(m_tr['value'].values)
    pw_m = np.array([0.5, 0.5, 1.0, 0.3, -3.0, 0.0, -0.2, 1.0, 0.7, np.log(1e6)])
    sw_m, ss_m = run_gibbs(Xm, ym, pw_m)
    m_thr = {k: float(np.percentile(m_tr['value'].values, p))
             for k, p in [('squad', 25), ('firstteam', 50), ('topflight', 75)]}

    all_train_vals = np.concatenate([f_tr['value'].values, m_tr['value'].values])
    hist_x_min = float(np.percentile(all_train_vals, 1))
    hist_x_max = float(np.percentile(all_train_vals, 99))

    f_full = pd.concat([pd.read_csv(dp('F_stats_values_train.csv')), pd.read_csv(dp('F_stats_values_test.csv'))], ignore_index=True)
    m_full = pd.concat([pd.read_csv(dp('M_stats_values_train.csv')), pd.read_csv(dp('M_stats_values_test.csv'))], ignore_index=True)
    m_full['league'] = m_full['league'].map(LEAGUE_MAP_M)
    f_full['date'] = pd.to_datetime(f_full['date'], errors='coerce')
    m_full['date'] = pd.to_datetime(m_full['date'], errors='coerce')
    f_cnt = f_full.groupby('player_id').size().rename('f_count')
    m_cnt = m_full.groupby('player_id').size().rename('m_count')
    f_lat = (f_full.sort_values('date').groupby('player_id', as_index=False)
             .last().assign(position='Forward'))
    m_lat = (m_full.sort_values('date').groupby('player_id', as_index=False)
             .last().assign(position='Midfielder'))
    combined = pd.concat([f_lat, m_lat], ignore_index=True)
    combined = combined.join(f_cnt, on='player_id').join(m_cnt, on='player_id')
    combined['f_count'] = combined['f_count'].fillna(0)
    combined['m_count'] = combined['m_count'].fillna(0)
    combined['preferred'] = combined.apply(
        lambda r: 'Forward' if r['f_count'] >= r['m_count'] else 'Midfielder', axis=1)
    pidx = (combined[combined['position'] == combined['preferred']]
            .drop_duplicates('player_id').copy())
    pidx['league_display'] = pidx['league'].map(lambda x: LEAGUE_DISPLAY.get(x, x))
    for col, cap in wcaps_f.items():
        pidx.loc[pidx['position'] == 'Forward', col] = \
            pidx.loc[pidx['position'] == 'Forward', col].clip(upper=cap)
    for col, cap in wcaps_m.items():
        pidx.loc[pidx['position'] == 'Midfielder', col] = \
            pidx.loc[pidx['position'] == 'Midfielder', col].clip(upper=cap)

    def pct_marks(df, col):
        return {'p25': float(df[col].quantile(0.25)),
                'p75': float(df[col].quantile(0.75)),
                'p90': float(df[col].quantile(0.90))}
    fw = pidx[pidx['position'] == 'Forward']
    mf = pidx[pidx['position'] == 'Midfielder']
    bmarks = {
        'Forward':    {c: pct_marks(fw, c) for c in perf_cols},
        'Midfielder': {c: pct_marks(mf, c) for c in perf_cols},
    }

    return dict(
        sw_f=sw_f, ss_f=ss_f, sw_m=sw_m, ss_m=ss_m,
        feat_f=feat_f, feat_m=feat_m,
        cmin_f=cmin_f, cmax_f=cmax_f, cmin_m=cmin_m, cmax_m=cmax_m,
        wcaps_f=wcaps_f, wcaps_m=wcaps_m,
        f_thr=f_thr, m_thr=m_thr,
        hist_x_min=hist_x_min, hist_x_max=hist_x_max,
        pidx=pidx, bmarks=bmarks,
        player_names=sorted(pidx['player_name'].dropna().unique().tolist()),
    )

D = load_and_fit()

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_x_row(position, xg, xa, xgc, age, year, league):
    caps = D['wcaps_f'] if position == 'Forward' else D['wcaps_m']
    cmin = D['cmin_f']  if position == 'Forward' else D['cmin_m']
    cmax = D['cmax_f']  if position == 'Forward' else D['cmax_m']

    def sc(v, col):
        return float(np.clip((min(v, caps[col]) - cmin[col]) /
                             (cmax[col] - cmin[col] + 1e-12), 0, 1))

    al = float(np.clip((np.log(max(age, 1)) - cmin['age_log']) /
                       (cmax['age_log'] - cmin['age_log'] + 1e-12), 0, 1))
    yr = float(np.clip((year - cmin['year']) /
                       (cmax['year'] - cmin['year'] + 1e-12), 0, 1))

    loh_f = {"EPL":[0,0,0,1],"La Liga":[0,0,1,0],"Bundesliga":[0,0,0,0],
             "Serie A":[1,0,0,0],"Ligue 1":[0,1,0,0]}
    loh_m = {"EPL":[0,0,1,0],"La Liga":[0,0,0,1],"Bundesliga":[0,0,0,0],
             "Serie A":[1,0,0,0],"Ligue 1":[0,1,0,0]}

    if position == 'Forward':
        return np.array([sc(xg,'xG_per_90'), sc(xa,'xA_per_90'), sc(xgc,'xGChain_per_90'),
                         al, yr] + loh_f[league] + [1.0])
    else:
        return np.array([sc(xg,'xG_per_90'), sc(xa,'xA_per_90'), sc(xgc,'xGChain_per_90'),
                         yr, al] + loh_m[league] + [1.0])

def posterior_draws(x_row, sw, ss, seed=42):
    np.random.seed(seed)
    return np.array([np.random.normal(x_row @ sw[j], np.sqrt(ss[j]))
                     for j in range(len(sw))])

def build_summary_text(position, median_val, lo_val, hi_val, mean_val):
    thr = D['f_thr'] if position == 'Forward' else D['m_thr']

    if median_val <= thr['squad']:
        tier_label = "a <strong>squad player</strong>"
        tier_desc  = ("falls in the lower quartile of the "
                      f"{position.lower()} market, typical of rotation options "
                      "or depth signings")
    elif median_val <= thr['firstteam']:
        tier_label = "a <strong>first-team asset</strong>"
        tier_desc  = ("sits around the median of the "
                      f"{position.lower()} market, representing a standard "
                      "first-team investment")
    elif median_val <= thr['topflight']:
        tier_label = "a <strong>top-flight talent</strong>"
        tier_desc  = ("falls in the upper quartile of the "
                      f"{position.lower()} market, an impactful starter "
                      "above average for their position")
    else:
        tier_label = "an <strong>elite player</strong>"
        tier_desc  = ("sits above the 75th percentile of the "
                      f"{position.lower()} market, a standout performer "
                      "commanding premium fees")

    rw = (hi_val - lo_val) / max(median_val, 1.0)
    if rw < 1.5:
        unc = (f"The 90% credible interval (€{lo_val/1e6:.1f}M to €{hi_val/1e6:.1f}M) "
               "is relatively tight, suggesting the model has comparatively high "
               "confidence in this range.")
    elif rw < 3.5:
        unc = (f"The 90% credible interval spans €{lo_val/1e6:.1f}M to €{hi_val/1e6:.1f}M, "
               "a moderate spread. Factors beyond statistics, such as contract length and "
               "injury history, likely account for the remaining uncertainty.")
    else:
        unc = (f"The 90% credible interval is wide (€{lo_val/1e6:.1f}M to €{hi_val/1e6:.1f}M), "
               "indicating this valuation is speculative. The median should be treated "
               "as a rough signal rather than a firm anchor.")

    skew = mean_val / max(median_val, 1.0)
    if skew > 1.25:
        snote = (f" The posterior mean (€{mean_val/1e6:.1f}M) sits above the median "
                 f"(€{median_val/1e6:.1f}M), reflecting right skew: a small probability "
                 "of very high valuations is pulling the average upward.")
    elif skew < 0.80:
        snote = (f" The posterior mean (€{mean_val/1e6:.1f}M) sits below the median "
                 f"(€{median_val/1e6:.1f}M), indicating mild left skew.")
    else:
        snote = ""

    return (f"<div class='val-summary'>"
            f"At a posterior median of <strong>€{median_val/1e6:.1f}M</strong>, "
            f"this player is estimated to be {tier_label}: their valuation "
            f"{tier_desc}. "
            f"{unc}{snote}"
            f"</div>")

def get_plot_cfg():
    is_dark = st.get_option("theme.base") == "dark"
    return dict(
        plot_bgcolor="#1a1a1a" if is_dark else "#ffffff",
        paper_bgcolor="#1a1a1a" if is_dark else "#ffffff",
        font=dict(family="IBM Plex Sans", size=12,
                  color="#f0f0f0" if is_dark else "#111111")
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.markdown("""
    <div class="hero">
      <h1>Football Transfer Value Forecasting with Uncertainty</h1>
      <p class="sub">LSTM performance forecasting + Bayesian linear regression · Forwards and Midfielders · 2014 – 2025</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([5, 4], gap="large")

    with col_l:
        st.markdown('<div class="slabel">The Problem</div>', unsafe_allow_html=True)
        st.markdown("""
The transfer market in professional football is one of the most prized and complex fields across
all sports. Each year clubs spend billions combined purchasing players, yet the process of valuing
them is largely driven by scouting opinions and market speculation rather than statistical analysis.
Clubs routinely overpay for players whose performance fails to justify their price tag, and
conversely overlook heavily undervalued talent. Beyond simply valuing players at the present
moment, clubs also face the challenge of anticipating how a player's value will evolve over time,
which depends on how their performance is likely to develop.

More formally, two related technical problems underlie these challenges. The first is a regression
problem: given a player's current performance statistics alongside metadata such as age, league and
year, can we reliably map those inputs to market value? The second is a forecasting problem: given
a player's historical performance statistics, can we predict how those statistics will evolve at
different points in the future? Each is valuable in isolation, but combining them yields a more
powerful framework capable of forecasting a player's market value at multiple future points.

This project builds a modular two-part framework to answer both.

        """)
        st.markdown("---")
        st.markdown('<div class="slabel">Data and Processing Pipeline</div>', unsafe_allow_html=True)
        for num, title, body in [
            ("Step 1", "Performance data via Understat API",
             "Per-match statistics collected across Europe's top-5 leagues (EPL, La Liga, Bundesliga, "
             "Serie A, Ligue 1) from 2014 onward. Understat records goals, assists, key passes, xG, "
             "xA, xGChain, and xGBuildup. Coverage is restricted to forwards and midfielders as "
             "Understat provides no meaningful defensive statistics."),
            ("Step 2", "Aggregate into 10-game blocks",
             "Rather than modelling noisy per-game figures, consecutive matches are grouped into "
             "rolling blocks of 10 games, roughly one quarter of a season. This smooths "
             "match-to-match variance while preserving granular form trends across a career."),
            ("Step 3", "Normalise to per-90 minutes",
             "All statistics are expressed per 90 minutes of playing time so blocks are directly "
             "comparable regardless of how many minutes were accumulated, which matters especially "
             "for frequent substitutes."),
            ("Step 4", "Feature selection: drop multicollinear statistics",
             "Of the five Understat metrics, xGBuildup and key passes were dropped due to "
             "significant multicollinearity with the retained features. Only xG, xA, and xGChain "
             "per 90 are used as performance inputs."),
            ("Step 5", "As-of merge with Transfermarkt valuations",
             "For each Transfermarkt valuation date, the most recent completed 10-game block prior "
             "to that date is matched in. Player age, calendar year (to capture "
             "transfer market inflation), and domestic league are appended as additional features."),
            ("Step 6", "80/20 player-based train and test split",
             "Players, not individual observations, are randomly assigned to train or test sets. "
             "This prevents the same player appearing in both splits, ensuring a fair evaluation "
             "on entirely unseen players."),
        ]:
            st.markdown(f"""<div class="step">
              <div class="snum">{num}</div><div class="stitle">{title}</div>
              <div class="sbody">{body}</div></div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="slabel">Two-Part Framework</div>', unsafe_allow_html=True)
        st.markdown("""
**Part 1: LSTM Performance Forecaster**

A Long Short-Term Memory (LSTM) neural network takes a sequence of 10 consecutive game blocks
as input, each a 3-vector of per-90 xG, xA, and xGChain, and predicts the next block.
Chaining predictions autoregressively allows multi-step forecasting. Trained separately for
forwards and midfielders via an 80/20 player-based split, with hyperparameters found by grid
search. Final config: learning rate 0.001, 1 hidden layer, width 32, 20 epochs. Test RMSE
was 0.224 for forwards and 0.123 for midfielders. The higher forward error likely reflects
heterogeneity in the position: strikers and wingers have very different play styles, and
the dataset does not provide granularity to separate them.

A known limitation is that autoregressive predictions collapse toward a player's
historical average rather than tracking their actual trajectory. This likely
relates to data limitations. LSTMs often need longer input sequences to capture
temporal patterns, but Understat does not provide enough data to produce enough
larger input sequences. Also, more independent and informative player inputs (beyond
those in the dataset) could be useful.

**Part 2: Bayesian Linear Regression**

Maps performance features (real or LSTM-generated) plus age, year, and one-hot league
encodings to log(market value) via Gibbs sampling with a Gaussian likelihood and informative
priors. To improve performance on underrepresented elite players, we oversample them during
training to increase representation. Observations with a market value exceeding 50M euros
were resampled with replacement to 3 times their original count. After 18% burn-in, 5,000
posterior samples are retained. Every valuation is a full posterior predictive distribution.
The model achieves a posterior R2 of 0.49 for midfielders and 0.40 for forwards using real
performance inputs, with RMSEs of 16.1M and 18.1M euros when fed real data. For LSTM inputs
it achieves a posterior R2 of 0.42 for midfielders and 0.34, with RMSEs of 22.5M and 24.3M euros
        """)
        st.markdown("---")
        st.markdown('<div class="slabel">Prior Weights and Posterior Results</div>', unsafe_allow_html=True)
        st.markdown("""
Priors encode domain knowledge: better performance should increase value, younger players command
a premium, and the Premier League carries the highest fees globally. Posteriors meaningfully
differ by position as anticipated. For midfielders, xGChain dominates with a posterior mean of
2.518, reflecting that general build-up participation is the most diagnostic statistic for that
position, while xG and xA carry much smaller weights (0.088 and 0.215). For forwards the picture
is more balanced: xG and xA reach 0.734 and 0.951, though xGChain still leads at 1.124, likely
driven by the creative contribution of wingers in the forward group.

The age coefficient is consistently negative at -3.902 for midfielders and -3.443 for forwards,
confirming that younger players command a substantial premium. Year is positive at 0.389 and
0.524, capturing transfer market inflation. The Premier League premium is the largest of all
league coefficients at 1.428 and 1.499, considerably higher than our prior and far ahead of
La Liga (0.663 / 0.758) and Serie A (0.508 / 0.595). Ligue 1 (0.389 / 0.524) is actually
higher than Bundesliga, which was the reference category, contrary to our prior of -0.2.
        """)
        st.markdown("---")
        st.markdown('<div class="slabel">Prior vs Posterior Weights</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Feature":        ["xG / 90","xA / 90","xGChain / 90","Age","Year",
                               "Premier League","La Liga","Ligue 1","Serie A","Intercept"],
            "FW Prior":       ["+1.5","+0.8","+0.4","-3.0","+0.3","+1.0","+0.7","-0.2","0.0","~13.8"],
            "FW Posterior": ["+0.593", "+0.814", "+0.944", "-3.443", "+0.524",
                 "+1.499", "+0.758", "+0.524", "+0.595", "~14.9"],
            "MF Prior":       ["+0.5","+0.5","+1.0","-3.0","+0.3","+1.0","+0.7","-0.2","0.0","~13.8"],
            "MF Posterior": ["+0.036", "+0.158", "+2.198", "-3.902", "+0.389",
                            "+1.428", "+0.663", "+0.389", "+0.508", "~14.9"],
        }), width='content', hide_index=True, height=388)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Value Estimator
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## Value Estimator")
    st.markdown(
        "Set a player's statistics and context below, then hit **Generate Valuation**. "
        "Optionally start from a known player: their stats will pre-fill the sliders "
        "so you can tweak individual inputs and see how the valuation shifts."
    )
    st.markdown("---")

    # ── Optional player prefill ────────────────────────────────────────────────
    st.markdown('<div class="slabel">Start from a player (optional)</div>', unsafe_allow_html=True)

    if 'search_reset' not in st.session_state:
        st.session_state['search_reset'] = 0

    col_search, col_clear = st.columns([5, 1])
    with col_search:
        selected = st.selectbox(
            "player_prefill",
            options=[""] + D['player_names'],
            format_func=lambda x: "Type a name to search..." if x == "" else x,
            label_visibility="collapsed",
            key=f"player_select_{st.session_state['search_reset']}",
        )
    with col_clear:
        if st.button("x Clear", width='content',
                     disabled=(selected == ""),
                     help="Clear player selection and return to manual entry"):
            st.session_state['search_reset'] += 1
            st.rerun()

    if selected:
        row        = D['pidx'][D['pidx']['player_name'] == selected].iloc[0]
        pf_pos     = row['position']
        pf_league  = LEAGUE_DISPLAY.get(row['league'], row['league'])
        pf_age     = int(round(row['age']))
        pf_xg      = float(row['xG_per_90'])
        pf_xa      = float(row['xA_per_90'])
        pf_xgc     = float(row['xGChain_per_90'])
        pf_year    = max(2015, min(2025, int(round(row['year']))
                        if not pd.isna(row['year']) else 2023))
        st.markdown(
            f"<div class='prefill-strip'>"
            f"<div class='ps-name'>{selected}</div>"
            f"<div class='ps-meta'>{pf_pos} · {pf_league} · "
            f"Age {pf_age} · {pf_year} season · "
            f"xG {pf_xg:.3f} · xA {pf_xa:.3f} · xGChain {pf_xgc:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        pf_pos    = "Forward"
        pf_league = "EPL"
        pf_age    = 23
        pf_xg     = None
        pf_xa     = None
        pf_xgc    = None
        pf_year   = 2023

    st.markdown("---")

    # ── Context inputs ─────────────────────────────────────────────────────────
    st.markdown('<div class="slabel">Player Context</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        position = st.radio("Position", ["Forward", "Midfielder"], horizontal=True,
                            index=0 if pf_pos == "Forward" else 1,
                            key=f"pos_{selected}_{st.session_state['search_reset']}")
    with c2:
        league = st.selectbox(
            "League", LEAGUES_UI,
            index=LEAGUES_UI.index(pf_league) if pf_league in LEAGUES_UI else 0,
            key=f"league_{selected}_{st.session_state['search_reset']}")
    with c3:
        age = st.slider("Age", 17, 38, pf_age,
                        key=f"age_{selected}_{st.session_state['search_reset']}",
                        help="Player's age at time of valuation.")
    with c4:
        year = st.slider("Year", 2015, 2025, pf_year,
                         key=f"year_{selected}_{st.session_state['search_reset']}",
                         help="Later years produce higher estimates due to market inflation.")

    # ── Stat caps and benchmarks ───────────────────────────────────────────────
    xg_cap, xa_cap, xgc_cap = (
        (0.99, 0.51, 1.34) if position == "Forward" else (0.67, 0.51, 1.21)
    )
    bm  = D['bmarks'][position]
    ref = D['pidx'][D['pidx']['position'] == position]

    def_xg  = float(np.clip(pf_xg  if pf_xg  is not None else ref['xG_per_90'].quantile(0.25),
                             0.0, xg_cap))
    def_xa  = float(np.clip(pf_xa  if pf_xa  is not None else ref['xA_per_90'].quantile(0.25),
                             0.0, xa_cap))
    def_xgc = float(np.clip(pf_xgc if pf_xgc is not None else ref['xGChain_per_90'].quantile(0.25),
                             0.0, xgc_cap))

    # ── Stat sliders ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slabel" style="margin-top:0.8rem">'
        'Performance Statistics: per 90 minutes across a 10-game block'
        '</div>', unsafe_allow_html=True
    )
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""<div class="stat-card">
          <span class="sc-name">Expected Goals (xG) / 90</span>
          <span class="sc-desc">Probability that the player&#39;s shots score, summed over the block.
            Measures goal-scoring threat, more stable than raw goal counts.</span>
          <div class="sc-benchmarks">
            <div class="sc-bmark">Bottom 25%: <span>{bm['xG_per_90']['p25']:.2f}</span></div>
            <div class="sc-bmark">Top 25%: <span>{bm['xG_per_90']['p75']:.2f}</span></div>
            <div class="sc-bmark">Top 10%: <span>{bm['xG_per_90']['p90']:.2f}</span></div>
          </div></div>""", unsafe_allow_html=True)
        xg = st.slider("xG/90", 0.0, xg_cap, def_xg, 0.01,
                       key=f"xg_{selected}_{position}_{st.session_state['search_reset']}",
                       label_visibility="collapsed")

    with s2:
        st.markdown(f"""<div class="stat-card">
          <span class="sc-name">Expected Assists (xA) / 90</span>
          <span class="sc-desc">xG value of passes that directly led to a shot.
            Measures creative contribution, chances created for teammates.</span>
          <div class="sc-benchmarks">
            <div class="sc-bmark">Bottom 25%: <span>{bm['xA_per_90']['p25']:.2f}</span></div>
            <div class="sc-bmark">Top 25%: <span>{bm['xA_per_90']['p75']:.2f}</span></div>
            <div class="sc-bmark">Top 10%: <span>{bm['xA_per_90']['p90']:.2f}</span></div>
          </div></div>""", unsafe_allow_html=True)
        xa = st.slider("xA/90", 0.0, xa_cap, def_xa, 0.01,
                       key=f"xa_{selected}_{position}_{st.session_state['search_reset']}",
                       label_visibility="collapsed")

    with s3:
        st.markdown(f"""<div class="stat-card">
          <span class="sc-name">xGChain / 90</span>
          <span class="sc-desc">Total xG of every possession the player touched that ended in
            a shot. Measures overall attacking involvement, especially important
            for midfielders.</span>
          <div class="sc-benchmarks">
            <div class="sc-bmark">Bottom 25%: <span>{bm['xGChain_per_90']['p25']:.2f}</span></div>
            <div class="sc-bmark">Top 25%: <span>{bm['xGChain_per_90']['p75']:.2f}</span></div>
            <div class="sc-bmark">Top 10%: <span>{bm['xGChain_per_90']['p90']:.2f}</span></div>
          </div></div>""", unsafe_allow_html=True)
        xgc = st.slider("xGChain/90", 0.0, xgc_cap, def_xgc, 0.01,
                        key=f"xgc_{selected}_{position}_{st.session_state['search_reset']}",
                        label_visibility="collapsed")

    st.markdown("")
    if st.button("Generate Valuation", type="primary"):
        sw = D['sw_f'] if position == 'Forward' else D['sw_m']
        ss = D['ss_f'] if position == 'Forward' else D['ss_m']
        xr = build_x_row(position, xg, xa, xgc, age, year, league)
        st.session_state['result'] = dict(
            position=position, league=league, age=age, year=year,
            xg=xg, xa=xa, xgc=xgc,
            log_draws=posterior_draws(xr, sw, ss),
            player_name=selected if selected else None,
        )

    # ── Results ────────────────────────────────────────────────────────────────
    if 'result' in st.session_state:
        r   = st.session_state['result']
        ld  = r['log_draws']
        vd  = np.exp(ld)
        pos = r['position']

        mean_v = np.exp(np.mean(ld))
        med_v  = np.exp(np.median(ld))
        lo_v   = np.exp(np.quantile(ld, 0.05))
        hi_v   = np.exp(np.quantile(ld, 0.95))

        st.markdown("---")
        if r['player_name']:
            st.markdown(f"#### Results for {r['player_name']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Posterior Mean",   f"€{mean_v/1e6:.1f}M")
        m2.metric("Posterior Median", f"€{med_v/1e6:.1f}M")
        m3.metric("90% CI Lower",     f"€{lo_v/1e6:.1f}M")
        m4.metric("90% CI Upper",     f"€{hi_v/1e6:.1f}M")

        st.markdown(build_summary_text(pos, med_v, lo_v, hi_v, mean_v),
                    unsafe_allow_html=True)

        st.markdown("### Posterior Predictive Distribution")
        st.markdown(
            "<span style='font-size:0.84rem;color:#555'>"
            "Each bar shows how many of the 5,000 Gibbs posterior samples landed in that "
            "value range. The x-axis is fixed to the full range of training market values, "
            "so the distribution visibly shifts left or right as you change inputs. "
            "Green bars fall within the 90% credible interval."
            "</span>", unsafe_allow_html=True
        )

        x_min = D['hist_x_min']
        x_max = D['hist_x_max'] * 1.1
        bins  = np.linspace(x_min, x_max, 70)
        bw    = (bins[1] - bins[0]) / 1e6

        counts, edges = np.histogram(vd, bins=bins)
        ci_msk = (edges[:-1] >= lo_v) & (edges[1:] <= hi_v)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=edges[:-1]/1e6, y=counts, width=bw,
            marker_color="#cccccc", name="Outside 90% CI"
        ))
        fig.add_trace(go.Bar(
            x=edges[:-1]/1e6, y=np.where(ci_msk, counts, 0), width=bw,
            marker_color="rgba(0,200,150,0.75)", name="90% Credible Interval"
        ))
        is_dark = (st.get_option("theme.base") or "light") == "dark"
        line_color = "#f0f0f0" if is_dark else "#0a0a0a"
        fig.add_vline(x=mean_v/1e6, line_color=line_color, line_dash="dash", line_width=2,
                      annotation_text=f"Mean €{mean_v/1e6:.1f}M",
                      annotation_position="top right", annotation_font_size=11, annotation_font_color=line_color)
        fig.add_vline(x=med_v/1e6, line_color="#e63946", line_dash="dot", line_width=2,
                      annotation_text=f"Median €{med_v/1e6:.1f}M",
                      annotation_position="top left", annotation_font_size=11, annotation_font_color="#e63946")
        
        text_color = "#f0f0f0" if is_dark else "#111111"
        grid_color = "#444444" if is_dark else "#eeeeee"

        fig.update_layout(
            **get_plot_cfg(),
            xaxis=dict(
                title="Market Value (€ millions)",
                title_font=dict(color=text_color),
                tickfont=dict(color=text_color),
                range=[x_min/1e6, x_max/1e6],
                gridcolor=grid_color,
            ),
            yaxis=dict(
                title="Samples",
                title_font=dict(color=text_color),
                tickfont=dict(color=text_color),
                gridcolor=grid_color,
            ),
            barmode="overlay",
            height=380, margin=dict(t=20, b=50),
        )
        st.plotly_chart(fig, width='content')