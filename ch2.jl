using Distributions
using StatsPlots

n = 9
w = 6

# Grid Approximation

p_grid = range(0, 1, length=20)
prior = fill(1., size(p_grid))
likelihood = [pdf(Binomial(n, p), w) for p in p_grid]
posterior = likelihood .* prior
posterior /= sum(posterior)

plot(p_grid, posterior)


# MCMC with Metropolis-Hastings algorithm (manual approach)

n_samples = 10000
samples = [0.5]
for i in 2:n_samples
    p_old = samples[i-1]
    p_new = rand(Normal(p_old, 0.1))
    p_new = p_new < 0 ? abs(p_new) : p_new
    p_new = p_new > 1 ? 2 - p_new : p_new
    q_old = pdf(Binomial(n, p_old), w)
    q_new = pdf(Binomial(n, p_new), w)
    push!(samples, rand() < q_new / q_old ? p_new : p_old)
end

density(samples)


# MCMC with Metropolis-Hastings algorithm (using Turing)

using Turing

@model function globetoss(n, w)
    p ~ Uniform(0, 1)
    w ~ Binomial(n, p)
    w, p
end

samples = sample(globetoss(n, w), MH(), 10000)


# Quadratic Approximation

using Optim
using LinearAlgebra

opt = optimize(globetoss(n, w), MAP())
var = Symmetric(informationmatrix(opt))
MvNormal(opt.values.array, var)
