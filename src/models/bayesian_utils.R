# This contains helpers for use in training a Bayesian either on midfielders or forwards.

library(scales)  # For Min-Max scaling
library(mvnfast) # For MVN calcs
library(MCMCpack) # For inverse gamma sampling
library(data.table) # For single feature one-hot encoding
library(ggplot2) # For plots

run_position_model <- function(position, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples=1000) {
    # This abstracts as much duplicate code as possible between the midfielder vs forward files
    
    league_levels <- c("Bundesliga", "Serie_A", "Ligue_1", "La_Liga", "EPL")
    
    # Data loading and prep
    res <- load_position_real_data(position)
    train_data <- one_hot_encode_league(res$train_data, league_levels)
    test_data  <- one_hot_encode_league(res$test_data,  league_levels)
    
    # Winsorize
    winsor_params <- read.csv(sprintf('src/data/%s_winsorize_params.csv', position))
    res <- winsorize(train_data, test_data, winsor_params)
    train_data <- res$train_data
    test_data <- res$test_data
    
    # Add squared age col
    train_data$age_log <- log(train_data$age)
    test_data$age_log <- log(test_data$age)
    
    # Get train/test matrices
    
    # Use these rather than indexing
    continuous_cols <- c('xG_per_90',
                         'xA_per_90',
                         'xGChain_per_90',
                         'age_log',
                         'year')
    
    drop_cols <- c("player_id", "player_name","age", "date", "value")
    
    # Store min/max from training
    col_mins <- apply(train_data[, continuous_cols], 2, min)
    col_maxs <- apply(train_data[, continuous_cols], 2, max)
    
    res <- get_X_y(train_data, continuous_cols, drop_cols, col_mins, col_maxs)
    X_train <- res$X
    y_train <- res$y
    
    res <- get_X_y(test_data, continuous_cols, drop_cols, col_mins, col_maxs)
    X_test <- res$X
    y_test <- res$y
    
    print(head(X_train))
    
    # Run Gibbs
    gibbs_res <- gibbs(X_train, y_train, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples=no_samples, burn=.18)
    
    # Return everything needed
    list(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,train_data=train_data,test_data=test_data,
         gibbs_samples_w=gibbs_res$w, gibbs_samples_sigmay=gibbs_res$sigmay, col_mins=col_mins, col_maxs=col_maxs)
}

evaluate_position_model <- function(position, res, player_name, real_ylim, lstm_ylim) {
    # This deduplicates the evaluation code between the midfielder vs forward files
    X_train <- res$X_train
    y_train <- res$y_train
    X_test <- res$X_test
    y_test <- res$y_test
    train_data <- res$train_data
    test_data <- res$test_data
    gibbs_samples_w <- res$gibbs_samples_w
    gibbs_samples_sigmay <- res$gibbs_samples_sigmay
    
    continuous_cols <- c('xG_per_90', 'xA_per_90', 'xGChain_per_90', 'age_log', 'year')
    drop_cols <- c("player_id", "player_name", "age", "date", "value")
    league_levels <- c("Bundesliga", "Serie_A", "Ligue_1", "La_Liga", "EPL")
    
    # check convergence and mixing of weights
    convergence_acf_plots(gibbs_samples_w)
    
    # mean weight
    cat("OLS:", solve(t(X_train)%*%X_train)%*%t(X_train)%*%y_train, "\n")
    cat("Posterior mean for w:", colMeans(gibbs_samples_w), "\n")
    
    plot(gibbs_samples_sigmay, type = "l", main="Convergence of sigmay samples")
    acf(gibbs_samples_sigmay, main="ACF of sigmay samples")
    
    ###################### Real stats
    no_samples <- nrow(gibbs_samples_w)
    
    y_test_pred <- get_preds(X_test, gibbs_samples_w, gibbs_samples_sigmay)
    cat("Real Stats\n")
    cat("RMSE:", rmse(exp(y_test), exp(y_test_pred)), "\n")
    cat("R^2:", R2_Score(y_pred = y_test_pred, y_true = y_test), "\n")
    
    ## Plot predictions vs actuals
    print(plot_preds_vs_actuals(y_test_pred, y_test, ylim = real_ylim,
                                title=sprintf("%s Predictions vs Actuals (Real data inputs)", position))) # Have to print else wont show
    cat("Predictions above 100mil:\n")
    print(test_data[exp(y_test_pred) > 100000000, ]) # Preds above 100mil
    
    # Look at preds for a player
    latest_idx   <- tail(which(test_data$player_name == player_name), 1) # Most recent
    player_preds <- player_posterior(X_test[latest_idx, ], gibbs_samples_w, gibbs_samples_sigmay)
    
    # Posterior value dist
    print(plot_player_posterior(X_test[latest_idx, ], player_name,
                          exp(y_test[latest_idx, ]),
                          gibbs_samples_w, gibbs_samples_sigmay, breaks = 20,
                          title=sprintf("Posterior Predictive Distribution (Real data input): %s", player_name)))
    
    cat(sprintf("90%% CI for %s:",player_name), exp(quantile(player_preds, c(0.1, 0.9))), "\n") # 90% ci
    
    # Change in val over time with credible int ribbon
    print(plot_player_value_over_time(player_name, X_test, y_test, test_data,
                                gibbs_samples_w, gibbs_samples_sigmay, ci = 0.9,
                                title=sprintf("%s Predictions vs Actuals over Time (Real data inputs)", player_name)))
    
    
    ##################### From LSTM
    full_test_data <- read.csv(sprintf("src/data/%s_predictions_real_values.csv",
                                       position))
    full_test_data <- one_hot_encode_league(full_test_data, league_levels)
    
    full_test_data$age_log <- log(full_test_data$age)
    
    col_mins <- res$col_mins
    col_maxs <- res$col_maxs
    
    ft_res <- get_X_y(full_test_data, continuous_cols, drop_cols, col_mins, col_maxs)
    X_full_test <- ft_res$X
    y_full_test <- ft_res$y
    
    y_full_test_pred <- get_preds(X_full_test, gibbs_samples_w, gibbs_samples_sigmay)
    cat("LSTM Predictions\n")
    cat("RMSE:", rmse(exp(y_full_test), exp(y_full_test_pred)), "\n")
    cat("R^2:", R2_Score(y_pred = y_full_test_pred, y_true = y_full_test), "\n")
    
    ## Plot predictions vs actuals
    print(plot_preds_vs_actuals(y_full_test_pred, y_full_test, ylim = lstm_ylim,
                                title=sprintf("%s Predictions vs Actuals (LSTM inputs)", position)))
    cat("Predictions above 100mil:\n")
    print(full_test_data[exp(y_full_test_pred) > 100000000, ]) # Preds above 100mil
    
    # Look at preds for a player
    latest_idx <- tail(which(full_test_data$player_name == player_name), 1)
    player_preds <- player_posterior(X_full_test[latest_idx, ], gibbs_samples_w, gibbs_samples_sigmay)
    
    # Posterior value dist
    print(plot_player_posterior(X_full_test[latest_idx, ], player_name,
                          exp(y_full_test[latest_idx, ]),
                          gibbs_samples_w, gibbs_samples_sigmay, breaks = 20,
                          title=sprintf("Posterior Predictive Distribution (LSTM input): %s", player_name)))
    
    
    cat(sprintf("90%% CI for %s:",player_name), exp(quantile(player_preds, c(0.1, 0.9))), "\n")
    
    # Change in val over time with credible int ribbon
    print(plot_player_value_over_time(player_name, X_full_test, y_full_test, full_test_data,
                                gibbs_samples_w, gibbs_samples_sigmay, ci = 0.9,
                                title=sprintf("%s Predictions vs Actuals over Time (LSTM inputs)", player_name)))
    
}

load_position_real_data <- function(position) {
    # This produces the train/test dataframes for real data at the given position
    train_data <- read.csv(sprintf("src/data/%s_stats_values_train.csv", position))
    test_data <- read.csv(sprintf("src/data/%s_stats_values_test.csv", position))
    list(train_data=train_data, test_data=test_data)
}

one_hot_encode_league <- function(data, league_levels) {
    # This one-hot encodes the league using the given league_levels for consistency
    
    # Force data to use same levels
    data$league <- factor(data$league, levels = league_levels)
    
    league_cols <- model.matrix(~ league - 1, data = data)[, -1]
    
    # Drop the original league column and add one-hot encoded version
    data <- cbind(data[, names(data) != "league"], league_cols)
    
    data
}

winsorize <- function(train_data, test_data, winsor_params) {
    # This winsorizes the given train and test dfs
    caps <- setNames(winsor_params$cap, winsor_params$stat)
    stat_cols <- names(caps)
    
    for (col in stat_cols) {
        train_data[, col] <- pmin(train_data[, col], caps[col])
        test_data[, col]  <- pmin(test_data[, col],  caps[col])
    }
    
    list(train_data=train_data, test_data=test_data)
}

get_X_y <- function(data, continuous_cols, drop_cols, col_mins, col_maxs) {
    # This helper produces the X and y matrices corresponding to the given data
    # It min/max scales X according to the given col min/max values, adds
    # a bias, and log transforms y.
    
    for (col in continuous_cols) {
        data[, col] <- (data[, col] - col_mins[col]) / (col_maxs[col] - col_mins[col])
    }
    
    X <- data.matrix(data[, !names(data) %in% drop_cols])
    y <- data.matrix(data[, "value"])
    
    # Log-transform y
    y <- log(y)
    
    # Add bias
    X <- cbind(X, bias = 1)
    
    list(X=X, y=y)
}

gibbs <- function(X_train, y_train, prior_w, prior_Sigma, prior_alpha, prior_beta, no_samples = 1000, burn=.18) {
    # Runs Gibbs sampling on the given X/y pair, and with the given prior choices
    # for the weights and uncertainties, as well as for the covariance of y.
    # Returns the sampled ws and Sigmas.
    p <- nrow(prior_w)
    n <- nrow(y_train)
    
    total_samples = floor(no_samples / (1 - burn))
    m = floor(total_samples * burn) # Discard amount
    
    # Set up markov chain matrices
    gibbs_samples_w <- matrix(0, nrow = total_samples, ncol = p)
    gibbs_samples_sigmay <- matrix(0, nrow = total_samples, ncol = 1)
    
    #initialize
    w_tilde <- matrix(0, nrow = p, ncol = 1)
    sigma2_tilde <- 1
    
    # run the sampler
    for (i in 1:total_samples) {
        # conditional posterior parameters for w
        V <- solve((1 / sigma2_tilde) * t(X_train) %*% X_train + solve(prior_Sigma))
        M <- V %*% ((1 / sigma2_tilde) * t(X_train) %*% y_train + solve(prior_Sigma) %*% prior_w)
        # sample weights
        w_tilde <- t(rmvn(1, M, V))
        
        # conditional posterior parameters for sigma2
        A <- (2 * prior_alpha + n) / 2
        B <- (2 * prior_beta + t(y_train - X_train %*% w_tilde) %*% (y_train - X_train %*% w_tilde)) / 2
        sigma2_tilde <- rinvgamma(1, A, B)
        
        # store them
        gibbs_samples_w[i, ] <- w_tilde
        gibbs_samples_sigmay[i, ] <- sigma2_tilde
    }
    
    list(w= gibbs_samples_w[(m+1):total_samples,], sigmay = gibbs_samples_sigmay[(m+1):total_samples,])
}

get_preds <- function(X, gibbs_samples_w, gibbs_samples_sigmay) {
    # Get test predictions
    test_preds <- matrix(0, nrow = nrow(X), ncol = nrow(gibbs_samples_w))
    for(i in 1:nrow(gibbs_samples_w)) {
        test_preds[,i] <- rmvn(n=1, mu=X%*%gibbs_samples_w[i,], sigma=diag(nrow(X))*gibbs_samples_sigmay[i])
    }
    rowMeans(test_preds)
}

rmse <- function(actuals, preds) {
    # Root mean squared error
    sqrt(mean((actuals - preds)^2))
}

player_posterior <- function(X_row, gibbs_samples_w, gibbs_samples_sigmay) {
    # Produces sampled log-value predictions for a given row.
    
    player_preds <- c()
    
    no_samples <- nrow(gibbs_samples_w)
    for(i in 1:no_samples){
        player_preds <- c(player_preds, rnorm(n = 1, mean = t(gibbs_samples_w[i,])%*%X_row, sd = sqrt(gibbs_samples_sigmay[i])))
    }
    
    player_preds
}

convergence_acf_plots <- function(samples) {
    # Convergence and ACF plots per feature
    for (i in 1:ncol(samples)) {
        plot(samples[,i], type = "l", main=sprintf("Convergence of feature %d",i))
        acf(samples[,i], main=sprintf("ACF of %d",i))
    }
}

# Plot predictions vs actuals - log scale each axis for readability
plot_preds_vs_actuals <- function(predictions,actuals,ylim,title) {
    # To euros
    pred_euros <- exp(predictions)
    actual_euros <- exp(actuals)
    
    ## GGplot much nicer
    df <- data.frame(actual = actual_euros, pred = pred_euros)
    
    ggplot(df, aes(x = actual, y = pred)) +
        geom_point(alpha = 0.3, size = 0.8, color = "blue") +
        geom_abline(aes(slope = 1, intercept = 0, linetype = "Perfect Prediction"),
                    color = "red", linewidth = 0.8) +
        scale_linetype_manual(name = "", values = c("Perfect Prediction" = "solid")) +
        scale_x_log10(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        scale_y_log10(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        labs(title = title,
             x = "Actual Value (euros)",
             y = "Predicted Value (euros)")
}

plot_player_posterior <- function(player_row, player_name, actual_value,
                                  gibbs_samples_w, gibbs_samples_sigmay, breaks,title) {
    # Get posterior samples
    preds <- player_posterior(player_row, gibbs_samples_w, gibbs_samples_sigmay)
    pred_euros <- exp(preds)
    
    # Use ggplot
    df <- data.frame(pred_euros = pred_euros)
    
    ggplot(df, aes(x = pred_euros)) +
        geom_histogram(bins = breaks, color="black") +
        geom_vline(aes(xintercept = actual_value,
                   colour = "Actual Value")) +
        scale_color_manual(name = "", values = c("Actual Value" = "red")) +
        scale_x_continuous(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        labs(title = title,
             x = "Predicted Market Value (euros)")
}

plot_player_value_over_time <- function(player_name, X, y, data,
                                        gibbs_samples_w, gibbs_samples_sigmay,
                                        ci = 0.9, title) {
    # Plots a player's predicted value over time with the credible interval bounds.
    
    # Get indices for this player
    player_mask <- data$player_name == player_name
    player_X <- X[player_mask, ]
    player_y <- exp(y[player_mask, ])
    player_dates <- as.Date(data$date[player_mask])
    
    upper_b <- ci + (1 - ci) / 2
    lower_b <- (1 - ci) / 2
    
    # Get posterior samples for each time point
    n_points <- sum(player_mask)
    
    means  <- numeric(n_points)
    lowers <- numeric(n_points)
    uppers <- numeric(n_points)
    
    for (i in 1:n_points) {
        preds <- exp(player_posterior(player_X[i, ], gibbs_samples_w, gibbs_samples_sigmay))
        means[i] <- mean(preds)
        lowers[i] <- quantile(preds, lower_b)
        uppers[i] <- quantile(preds, upper_b)
    }
    
    # Make plot
    df <- data.frame(
        date = player_dates,
        mean = means,
        actual = player_y,
        lower = lowers,
        upper = uppers,
        ci_label = sprintf("%d%% Credible Interval", 100 * ci)
    )
    
    ggplot(df, aes(x = date)) +
        geom_ribbon(aes(ymin = lower, ymax = upper, fill = ci_label)) +
        geom_line(aes(y = mean, color="Predicted Mean"), linewidth = 1) +
        geom_line(aes(y = actual, color = "Actual Value"), linewidth = 1) +
        scale_fill_manual(name = "", values = setNames("lightblue", df$ci_label[1])) +
        scale_color_manual(name = "", values = c("Predicted Mean" = "blue", "Actual Value" = "red")) +
        scale_y_continuous(labels = function(x) formatC(x, format = "f", digits = 0, big.mark = ",")) +
        labs(title=title,
            x= "Date",
             y = "Market Value (euros)")
}