# This contains code to fit and evaluate a Bayesian model on midfielders.
source('src/models/bayesian_utils.R')

library(MLmetrics) # Contains R^2 function
set.seed(42) # for reproducibility

# CONFIG INFO
position <- 'M'

# Define priors
prior_w <- matrix(c(0.2, # xg - small positive since goals aren't so important for M
                    0.2, # xa - same idea as for xg
                    2, # xgchain - midfielders make a larger contribution in buildup
                    0.3, # year - small positive for inflation
                    -3, # age - strong negative
                    0, # Serie A - equivalent to Bundesliga
                    -0.2, # Ligue 1 - worse than Bunesliga
                    1, # PL - most expensive league
                    .7, # La liga - next most after PL
                    log(1000000) # intercept - when everything else is 0 (min stats + Bundesliga)
), ncol = 1)
p <- nrow(prior_w)
prior_Sigma <- diag(p)*1
# Now need priors for sigma_y
prior_alpha <- 1
prior_beta <- 1

no_samples <- 100

# Preprocess + run Gibbs
res <- run_position_model(position, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples=no_samples)
# Evaluate
evaluate_position_model(position, res, player_name="Bukayo Saka", real_ylim = c(0, 2e8), lstm_ylim=c(0,5e7))