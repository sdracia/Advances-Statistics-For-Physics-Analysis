data {
  int n;  // Total number of samples
  int y;  // Number of samples with high bacter X level
}

parameters {
  real p;  // Probability of high bacter X level
}

model {
  p ~ beta(1, 10);  // Beta(1, 10) prior for p
  y ~ binomial(n, p);  // Likelihood function
}
