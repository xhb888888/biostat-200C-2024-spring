data {
  int<lower=0> N;         // number of data points
  int<lower=0> K;         // number of predictors
  int<lower=0> J;         // number of groups
  int<lower=0, upper=1> y[N]; // response variable
  matrix[N, K] x;         // predictor matrix
  int<lower=1, upper=J> id[N]; // group identifier
}
parameters {
  vector[K] beta;         // coefficients for predictors
  real alpha[J];          // group-specific intercepts
  real<lower=0> sigma;    // standard deviation of group intercepts
}
model {
  alpha ~ normal(0, sigma);          // prior for group intercepts
  beta ~ normal(0, 10);              // prior for coefficients
  sigma ~ cauchy(0, 2.5);            // prior for sigma
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(alpha[id[n]] + dot_product(row(x, n), beta)); // likelihood
  }
}