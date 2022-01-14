using Distributions
using StatsPlots
using StatsBase
using Turing

@model function globetoss(n, w)
    p ~ Uniform(0, 1)
    w ~ Binomial(n, p)
    w, p
end

samples1 = sample(globetoss(9, 6), MH(), 10000)

plot(samples1)

mean(samples1[:p] .< 0.5)
mean(0.5 .< samples1[:p] .< 0.75)
quantile(samples1; q=[0.8])
quantile(samples1; q=[0.1, 0.9])

preds1 = predict(globetoss(9, missing), samples1)
mixeddensity(preds1)

samples2 = sample(globetoss(3, 3), MH(), 10000)

plot(samples2)

quantile(samples2; q=[0.25, 0.75])
hpd(samples2; alpha=0.5)

preds2 = predict(globetoss(3, missing), samples2)
mixeddensity(preds2)


