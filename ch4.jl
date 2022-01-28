using CSV
using DataFrames
using Turing
using StatsPlots


@model function dobe_!kung_stature(w, h, h̄)
    α ~ Normal(60, 10)
    β ~ LogNormal(0, 1)
    σ ~ Uniform(0, 50)
    μ = α .+ β .* (h .- h̄)
    w ~ MvNormal(μ, σ)
end

# Simulated data

α = 70
β = 0.5
σ = 5
n = 100
h = rand(Uniform(130, 170), n)
μ = α .+ β * (h .- mean(h))
w = rand(MvNormal(μ, σ))

model = dobe_!kung_stature(w, h, mean(h))
post = sample(model, NUTS(), 10000)


# Dobe !Kung data collected by Nancy Howell in 1969

howell = CSV.read(joinpath("data", "howell1.txt"), DataFrame; delim=';')
howell = howell[howell.age .>= 18, :]
plot(howell.height, howell.weight, seriestype=:scatter, legend=false, alpha=0.7)

mean_howell_height = mean(howell.height)
model = dobe_!kung_stature(howell.weight, howell.height, mean_howell_height)
post = sample(model, NUTS(), 10000)
plot(post)


μ_at_160 = post[:α] + post[:β] * (160 - mean_howell_height)

density(μ_at_160, legend=false)
quantile(vec(μ_at_160), [0.055, 0.945])

height_seq = range(extrema(howell.height)..., length=100)
μ_post = post[:α] .+ post[:β] * (height_seq .- mean_howell_height)'
μ_ci = [quantile(μ_post[:, i], [0.055, 0.945]) for i in 1:size(μ_post, 2)]
μ_mean = vec(mean(μ_post, dims=1))

plot(repeat(height_seq, length(height_seq)), vec(μ_post'), seriestype=:scatter, legend=false, alpha=0.1)

plot(howell.height, howell.weight, seriestype=:scatter, legend=false, alpha=0.6)
plot!(height_seq, μ_mean, ribbon=(μ_mean .- first.(μ_ci), last.(μ_ci) .- μ_mean))

weight_sim = predict(dobe_!kung_stature(missing, height_seq .- mean_howell_height), post)
weight_pi = [quantile(vec(weight_sim[name]), [0.055, 0.945]) for name in names(weight_sim)]

plot(howell.height, howell.weight, seriestype=:scatter, legend=false, alpha=0.6)
plot!(height_seq, μ_mean, ribbon=(μ_mean .- first.(μ_ci), last.(μ_ci) .- μ_mean), fillalpha=0.4)
plot!(height_seq, μ_mean, ribbon=(μ_mean .- first.(weight_pi), last.(weight_pi) .- μ_mean), linewidth=0, fillalpha=0.4)


@model function dobe_!kung_stature2(w, h, h̄, s)
    ν ~ filldist(Normal(160, 10), 2)
    τ ~ Uniform(0, 10)
    h ~ MvNormal(ν[s], τ)

    α ~ filldist(Normal(60, 10), 2)
    β ~ filldist(LogNormal(0, 1), 2)
    σ ~ Uniform(0, 10)
    μ = α[s] .+ β[s] .* (h .- h̄)
    w ~ MvNormal(μ, σ)
end

model = dobe_!kung_stature2(howell.weight, howell.height, mean_howell_height, howell.male .+ 1)
post = sample(model, NUTS(), 10000)

# Causal effect of S on W?

h_s1 = rand.(Normal.(post["ν[1]"], post["τ"]))
w_s1 = rand.(Normal.(post["α[1]"] .+ post["β[1]"] .* (h_s1 .- mean_howell_height), post["σ"]))

h_s2 = rand.(Normal.(post["ν[2]"], post["τ"]))
w_s2 = rand.(Normal.(post["α[2]"] .+ post["β[2]"] .* (h_s2 .- mean_howell_height), post["σ"]))

w_do_s = w_s2 - w_s1
density(w_do_s)

# _Direct_ causal effect of S on W?

h_seq = range(130, 190, length=50)
μ_s1 = post["α[1]"] .+ post["β[1]"] .* h_seq'
μ_s2 = post["α[2]"] .+ post["β[2]"] .* h_seq'
μ_diff = μ_s1 - μ_s2

μ_diff_50 = [quantile(μ_diff[:, i], [0.25, 0.75]) for i in 1:size(μ_diff, 2)]
μ_diff_75 = [quantile(μ_diff[:, i], [0.125, 0.875]) for i in 1:size(μ_diff, 2)]
μ_diff_90 = [quantile(μ_diff[:, i], [0.05, 0.95]) for i in 1:size(μ_diff, 2)]
μ_diff_mean = vec(mean(μ_diff, dims=1))

plot(h_seq, μ_diff_mean, ribbon=(μ_diff_mean .- first.(μ_diff_90), last.(μ_diff_90) .- μ_diff_mean), fillalpha=0.4, legend=false, linewidth=0)
plot!(h_seq, μ_diff_mean, ribbon=(μ_diff_mean .- first.(μ_diff_75), last.(μ_diff_75) .- μ_diff_mean), fillalpha=0.4, legend=false, linewidth=0)
plot!(h_seq, μ_diff_mean, ribbon=(μ_diff_mean .- first.(μ_diff_50), last.(μ_diff_50) .- μ_diff_mean), fillalpha=0.4, legend=false, linewidth=0)
plot!(h_seq, fill(0, length(h_seq)), line=:dash, legend=false, color="white")
