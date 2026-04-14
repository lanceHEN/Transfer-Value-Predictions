# This contains code to fit and evaluate a Bayesian model on midfielders.
source('src/models/bayesian_utils.R')

library(MLmetrics) # Contains R^2 function
set.seed(42) # for reproducibility

# CONFIG INFO
position <- 'M'

# Define priors
prior_w <- matrix(c(0.5, # xg - small positive since goals aren't so important for M
                    0.5, # xa - same idea as for xg
                    1, # xgchain - midfielders make a larger contribution in buildup
                    -3, # age - strong negative
                    0.3, # year - small positive for inflation
                    0, # Serie A - equivalent to Bundesliga
                    -0.2, # Ligue 1 - worse than Bunesliga
                    .4, # La liga - next most after PL
                    .7, # PL - most expensive league
                    log(1000000) # intercept - when everything else is 0 (min stats + Bundesliga)
), ncol = 1)
p <- nrow(prior_w)
prior_Sigma <- diag(p)*0.01
# Now need priors for sigma_y
prior_alpha <- 1
prior_beta <- 1

no_samples <- 5000

# Preprocess + run Gibbs
res <- run_position_model(position, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples=no_samples, oversample_elite=T)
# Evaluate
evaluate_position_model(position, res, player_name="Bukayo Saka")