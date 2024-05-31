data {
  int N;
  array[N] int y;
}
parameters {
  real lambda;
}
model {
  lambda ~ beta(1, 1);
  y ~ poisson(lambda);
}
