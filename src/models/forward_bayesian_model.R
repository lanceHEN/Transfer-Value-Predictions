# This contains code to fit and evaluate a Bayesian model on forwardss
source('src/models/bayesian_utils.R')

library(MLmetrics) # Contains R^2 function
set.seed(42) # for reproducibility

# CONFIG INFO
position <- 'F'

# Define priors
prior_w <- matrix(c(0.5, # xg - larger since forwards should score more goals
                    0.5, # xa - larger since forwards should get more assists
                    1, # xgchain - not as great as for midfielders
                    -3, # age^2 - strong negative
                    0.3, # year - small positive for inflation
                    0, # Serie A - equivalent to Bundesliga
                    -0.2, # Ligue 1 - worse than Bunesliga
                    .7, # La liga - next most after PL
                    1, # PL - most expensive league
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
evaluate_position_model(position, res, player_name="Mason Greenwood", real_ylim = c(0, 2e8), lstm_ylim=c(0,5e7))
