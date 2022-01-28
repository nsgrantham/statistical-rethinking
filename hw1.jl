using Distributions  # v0.25
using StatsPlots     # v0.14
using Turing         # v0.19

# 1. Suppose the globe tossing data (Chapter 2) had turned out to be
# 4 water and 11 land. Construct the posterior distribution, using
# grid approximation. Use the same flat prior as in the book.

n = 15
w = 4
p_grid = range(0, 1, length=1000)
prior_grid = pdf.(Uniform(0, 1), p_grid)
like_grid = pdf.(Binomial.(n, p_grid), w)
post_grid = like_grid .* prior_grid
post_grid /= sum(post_grid)
plot(p_grid, post_grid, legend=false)

# 2. Now suppose the data are 4 water and 2 land. Compute the posterior
# again, but this time use a prior that is zero below p = 0.5 and a
# constant above p = 0.5. This corresponds to prior information that
# a majority of the Earth's surface is water.

n = 6
w = 4
prior_grid = pdf.(Uniform(0.5, 1), p_grid)
like_grid = pdf.(Binomial.(n, p_grid), w)
post_grid = like_grid .* prior_grid
post_grid /= sum(post_grid)
plot(p_grid, post_grid, legend=false)

# 3. For the posterior distribution from 2, compute 89% percentile and
# HPDI intervals. Compare the widths of these intervals. Which is wider?
# Why? If you had only the information in the interval, what might you
# misunderstand about the shape of the posterior distribution?

@model function globetoss(n, w)
    p ~ Uniform(0.5, 1.)
    w ~ Binomial(n, p)
    w, p
end

post = sample(globetoss(n, w), MH(), 10000)

traceplot(post)
density(post)

quantile(post, q=[0.055, 0.945])
hpd(post, alpha=0.11)

# The 89% percentile interval is wider than the HPD interval because
# the HPD interval is, by definition, the narrowest region with
# 89% of the posterior probability. If we only had information in the
# interval, we might completely miss the fact that the distribution
# is truncated on 0.5 to 1.0.

# 4. OPTIONAL CHALLENGE. Suppose there is bias in sampling so that
# Land is more likely than Water to be recorded. Specifically, assume
# that 1-in-5 (20%) of Water samples are accidentally recorded instead
# as Land. First, write a generative simulation of this sampling process.
# Assuming the true proportion of Water is 0.70, what proportion does
# your simulation tend to produce instead? Second, using a simulated
# sample of 20 tosses, compute the unbiased posterior distribution of
# the true proportion of Water.

# Simulation will tend to produce 0.8 * 0.7 = 0.56 proportion of Water.

BiasedBinomial(n, p) = Binomial(n, 0.8p)

@model function globetoss(n, w)
    p ~ Uniform(0, 1)
    w ~ BiasedBinomial(n, p)
end

n = 20
w = rand(BiasedBinomial(n, 0.7))

post = sample(globetoss(n, w), MH(), 10000)

traceplot(post)
density(post)
