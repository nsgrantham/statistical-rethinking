using CSV
using DataFrames
using StatsPlots
using Turing


# 1. Construct a linear regression of weight as predicted by height,
# using the adults (age 18 or greater) from the Howell1 dataset. The
# heights listed below were recorded in the !Kung census, but weights
# were not recorded for these individuals. Provide predicted weights
# and 89% compatibility intervals for each of these individuals. That
# is, fill in the table below, using model-based predictions.

# | i | height | expected weight | 89% interval |
# |---|--------|-----------------|--------------|
# | 1 |   140  |                 |              |
# | 2 |   160  |                 |              |
# | 3 |   175  |                 |              |

howell = CSV.read(
    download("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"),
    DataFrame,
    delim=';'
)
adult = howell[howell.age .>= 18, :];
plot(adult.height, adult.weight, ylab="Adult Weight (kg)", xlab="Adult Height (cm)", seriestype=:scatter, legend=false)

@model function adult_stature(w, h, h̄)
    α ~ Normal(50, 10)
    β ~ LogNormal(0, 1)
    σ ~ Exponential(1)
    μ = α .+ β .* (h .- h̄)
    w ~ MvNormal(μ, σ)
end

model = adult_stature(adult.weight, adult.height, mean(adult.height))
post = sample(model, NUTS(), 10_000)
preds = predict(adult_stature(missing, [140, 160, 175], mean(adult.height)), post)

mean(preds)  # expected weights
quantile(preds, q=[0.055, 0.945])  # 89% intervals


# 2. From the Howell1 dataset, consider only the people younger than 13 years
# old. Estimate the causal association between age and weight. Assume that
# age influences weight through two paths. First, age influences height,
# and height influences weight. Second, age directly influences weight
# through age-related changes in muscle growth and body proportions. All
# of this implies this causal model (DAG):
#
#       H
#      ^ \
#     /   \
#    /     v
#   A ----> W
#
# Use a linear regression to estimate the total (not just direct) causal
# effect of each year of growth on weight. Be sure to carefully consider the
# priors. Try using prior predictive simulation to assess what they imply.

@model function child_stature(w, a)
    α ~ Normal(4, 1)
    β ~ LogNormal(0, 0.5)
    σ ~ Exponential(0.5)
    μ = α .+ β .* a
    w ~ MvNormal(μ, σ)
end

child = howell[howell.age .< 13, :]
model = child_stature(child.weight, child.age)

prior = sample(model, Prior(), 1_000)
age_seq = collect(0:12)
prior_preds = predict(child_stature(missing, age_seq), prior)
prior_preds = dropdims(prior_preds.value, dims=3)'
plot(repeat(age_seq, size(prior_preds, 2)), vec(prior_preds), ylab="Child Weight (kg)", xlab="Child Age", title="Prior Predictive Simulations", alpha=0.1, seriestype=:scatter, legend=false)

plot(child.age, child.weight, ylab="Child Weight (kg)", xlab="Child Age", title="Howell's Dobe !Kung Census", seriestype=:scatter, legend=false)

Now we fit a linear regression model to the data and plot the marginal posterior distribution of β, which represents the effect of each year of growth on child weight.

model = child_stature(child.weight, child.age);
post = sample(model, NUTS(), 10_000)
density(post["β"], ylab="Density", xlab="β", legend=false)


# 3. Now suppose the causal association between age and weight might be
# different for boys and girls. Use a single linear regression, with a
# categorical variable for sex, to estimate the total causal effect of age
# on weight separately for boys and girls. How do girls and boys differ?
# Provide one or more posterior contrasts as a summary.

#
#       H <---- S
#      ^ \     /
#     /   \   /
#    /     v v
#   A ----> W
#

@model function child_stature(w, a, s)
    α ~ filldist(Normal(4, 1), 2)
    β ~ filldist(LogNormal(0, 0.5), 2)
    σ ~ Exponential(0.5)
    μ = α[s] .+ β[s] .* a
    w ~ MvNormal(μ, σ)
end

model = child_stature(child.weight, child.age, child.male .+ 1)
post = sample(model, NUTS(), 10_000)

age_seq = range(0, 12, length=50)
w_f = predict(child_stature(missing, age_seq, 1), post)
w_m = predict(child_stature(missing, age_seq, 2), post)
w_diff = dropdims(w_m.value .- w_f.value, dims=3)'

w_diff_mean = vec(mean(w_diff, dims=2))
w_diff_50 = mapslices(x -> quantile(x, [0.25, 0.75]), w_diff, dims=2)
w_diff_70 = mapslices(x -> quantile(x, [0.15, 0.85]), w_diff, dims=2)
w_diff_90 = mapslices(x -> quantile(x, [0.05, 0.95]), w_diff, dims=2)

plot(age_seq, w_diff_mean, ribbon=abs.(w_diff_mean .- w_diff_90), ylab="Male - Female Child Weight Difference (kg)", xlab="Child Age")
plot!(age_seq, w_diff_mean, ribbon=abs.(w_diff_mean .- w_diff_70), legend=false)
plot!(age_seq, w_diff_mean, ribbon=abs.(w_diff_mean .- w_diff_50), legend=false)
plot!(age_seq, fill(0, length(age_seq)), line=:dash, color=:white, legend=false)


# 4. OPTIONAL CHALLENGE. The data in `Oxboys` are growth records for 26 boys
# measured over 9 periods. I want you to model their growth. Specifically,
# model the increments in growth from one period (Occasion in the data table)
# to the next. Each increment is simply the difference between height in one
# occasion and height in the previous occasion. Since none of these boys shrunk
# during the study, all of the growth increments are greater than zero. Estimate
# the posterior distribution of these increments. Constrain the distribution so
# it is always positive — it should not be possible for the model to think that
# boys can shrink from year to year. Finally compute the posterior distribution
# of the total growth over all 9 occasions.

oxboys = CSV.read(
    download("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Oxboys.csv"),
    DataFrame,
    delim=';'
)
oxboys = combine(groupby(oxboys, :Subject), :height => diff => :diff)
oxboys = transform(groupby(oxboys, :Subject), :Subject => (x -> 1:length(x)) => :time)

@model function oxboy_growth(d, t)
    μ ~ filldist(Normal(0, 0.2), length(unique(t)))
    σ ~ Exponential(0.2)
    d ~ MvLogNormal(μ[t], σ)
end

prior = sample(oxboy_growth(oxboys.diff, oxboys.time), Prior(), 1_000)
prior_preds = predict(oxboy_growth(missing, 1:8), prior)
prior_preds = dropdims(prior_preds.value, dims=3)'
plot(repeat(1:8, size(prior_preds, 2)), vec(prior_preds), ylab="Height increment (cm)", xlab="Time", title="Prior Predictive Simulations", alpha=0.1, seriestype=:scatter, legend=false)

plot(oxboys.time, oxboys.diff, ylab="Height increment (cm)", xlab="Time", seriestype=:scatter, legend=false)

model = oxboy_growth(oxboys.diff, oxboys.time)
post = sample(model, MH(), 10_000)

d_sim = predict(oxboy_growth(missing, 1:8), post)
d_sim = dropdims(d_sim.value, dims=3)'
d_mean = vec(mean(d_sim, dims=2))
d_90 = mapslices(x -> quantile(x, [0.05, 0.95]), d_sim, dims=2)

plot(oxboys.time, oxboys.diff, ylab="Height increment (cm)", xlab="Time", seriestype=:scatter, alpha=0.3, legend=false)
plot!(1:8, d_mean, seriestype=:scatter, yerror=(d_mean .- d_90), markersize=8, legend=false)

growth_sim = vec(sum(d_sim, dims=1))
density(growth_sim, ylab="Density", xlab="Total growth (cm)", legend=false)
