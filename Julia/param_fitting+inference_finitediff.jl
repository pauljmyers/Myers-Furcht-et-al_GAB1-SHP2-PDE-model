## =============== Activate project environment =============== ##
#= 
Store current working directory where this script and the Project.toml and Manifest.toml
files for this project are located: 
=#
cwd = pwd() # Get current directory

# -- Activate and precompile the project's environment:
using Pkg
Pkg.activate(cwd)
# Pkg.instantiate()     # installs all package dependencies for the current project/environment
# Pkg.precompile()    # may take a few minutes to complete


## =============== Using Distributed parallelism? =============== ##
# -- Set number of workers for Distributed:
n_workers = 5

# -- Use Distributed parallelism?
use_distributed = false
using Distributed
if use_distributed
    if nprocs() == 1
        # addprocs(n_workers, exeflags="--project") # adds the same number of processes as available CPU cores
        addprocs(n_workers, exeflags="--project=$(Base.active_project())") # adds the same number of processes as available CPU cores
    end
    display(nprocs())    # check number of available processes
    display(nworkers())  # check number of available workers
end



## =============== Load Julia libraries =============== ##
@everywhere begin 
    using Turing, NumericalIntegration
    using CSV, DataFrames, DataFramesMeta, Pipe, StatsBase
    using Measurements  # for automatic propagation of uncertainties using the `Measurement` type
    using Random
end
using Interpolations
using CairoMakie, AlgebraOfGraphics
using Dates
using JLD2
const mk = Makie
const aog = AlgebraOfGraphics

currentday = string(today())
currenttime = replace(string(Time(Dates.format(Dates.now(),"HH:MM"))), ":"=>"-")

## =============== Load PDE system and functions for generating priors =============== ##
# @everywhere include("basepdesolver.jl")     # file with model equations
# -- Define PDESystem and load necessary solver functions → may take several minutes
@everywhere include("basepdesolver.jl")  # change problem setup (e.g., r and t vectors) in this script    


## =============== Load dynamic model data for comparison =============== ##
# -- Exptl percent SHP2-bound GAB1 @ 15 min EGF (from Furcht et al., Sci Sig 2015, Fig. 5D):
fitdata_df = DataFrame(μ=26.426, σ=5.4059*sqrt(3))  
CSV.write("exptl_pct_SHP2-bound-GAB1.csv", fitdata_df)


## =============== Define model parameters, variables, and domains =============== ##
# -- Grid/time bounds:
@everywhere begin
    dr = 0.2 # length of each spatial step
    R = 10.0 #  Radius of cell
    Nts = 100 #  Number of time points to store
    tf = 5.0 # final integration time
    tvec = collect(0:tf/Nts:tf) # time vector
    volCF = 1/(4/3*pi*R^3) # conversion factor from cell⁻¹ to 1/μm³ units for a spherical cell
    surfCF = 1/(4*pi*R^2) # conversion factor from cell⁻¹ to 1/μm² units for a spherical cell
end

# -- Diffusivities (μm²/min):
D_S = 82.0     # SFKs
D_G2 = 130.0   # GRB2
D_G2G1 = 61.0  # GRB2-GAB1
D_G2G1S2 = 55.0 # GRB2-GAB1-SHP2
D_G1 = 66.0    # GAB1
D_G1S2 = 56.0  # GAB1-SHP2
D_S2 = 78.0    # SHP2

# -- EGF concentration:
@everywhere EGF = 1.67e-3  # EGF concentration, μM -- equivalent to 10 ng/mL

# -- Define initial concentrations in molecules/μm³ or molecules/μm² (for surface species):
@everywhere begin 
    CoSFK = 6.0e5*volCF  # Total concentration of SFK (molec/μm³)
    CoG2 = 6.0e5*volCF   # Total concentration of GRB2 (molec/μm³)
    CoG1 = 6.0e5*volCF   # Total concentration of GAB1 (molec/μm³)
    CoS2 = 6.0e5*volCF   # Total concentration of SHP2 (molec/μm³)
    CoEGFR = 6.0e5*surfCF    # Total concentration of EGFR (molec/μm²)
    Co = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations
end

## Get parameter prior distributions:
#= The key variables defined w/ this script below are `uvpars` and `mvpars`, as well as vectors
containing the diffusivities and kinetic parameters (`Diffs` and `kvals`). This also defines
the EGF concentration for model simulations (10 ng/mL, or 1.67e-3 µM).
=#
include("get_param_priors.jl")


# -- Define parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]
pnames = [Dnames; knames]
Cnames = ["CoSFK", "CoG2", "CoG1", "CoSHP2", "CoEGFR"]   # Names of initial concentration variables


## ===== Get lognormal params for experimental fitting data ===== ##
fitdata_lognorm = log.(calcModeSpread([fitdata_df.μ[1] fitdata_df.σ[1] 1.0 0.0]))
fitdata_ln_dist = truncated(LogNormal(fitdata_lognorm[1], fitdata_lognorm[2]), upper=100.0)

plot(fitdata_ln_dist, 
    axis=(;xlabel="% SHP2-bound GAB1", ylabel="density", title="Experimental data",
        width=300,height=300,xticks=0:10:100)
    ) |> display



        
## =============== Call PDE solver =============== ##
@time pdesolver_fitting([pvals; Co]; R=R, dr=dr, tf=tf, Nts=Nts, tol=1e-3, maxiters=20)




## Test autodiff:
using ForwardDiff
function testf(x::AbstractVector{T}; 
    Co=Co, CoG1=Co[3], Diffs=Diffs, kvals=kvals, D_inds=D_inds, k_inds=k_inds, sa=1/surfCF, vol=1/volCF) where T
    kvals = T[kvals...]
    Diffs = T[Diffs...]
    x2 = copy(kvals)
    x2[6:9] .= x

    sol, r_sol = pdesolver_fitting([Diffs; x2; Co]; R=R, dr=dr, tf=tf, Nts=Nts, tol=1e-2)

    PG1S_cyt = sol.PG1S[:,end] .+ sol.G2PG1S[:,end]   # cytoplasmic GAB1-SHP2
    PG1S_cyt_ave = NumericalIntegration.integrate(r_sol, PG1S_cyt.*r_sol.^2) .* 3.0./R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_mem = sol.EG2PG1S[end] .* sa/vol     # membrane GAB1-SHP2
    PG1Stot = PG1S_cyt_ave .+ PG1S_mem     # total GAB1-SHP2
    pct_bound = PG1Stot[1]/CoG1*100.0    # model-predicted percent SHP2-bound GAB1
    return pct_bound
end


##
@time ForwardDiff.gradient(testf, kvals[6:9])













## ============== Initial parameter fitting with Optimization.jl =============== ##
# -- Load Optimization.jl packages:
using Optimization
using OptimizationNLopt
using OptimizationMultistartOptimization
# using OptimizationMetaheuristics


# -- Select model parameters to fit and set their initial values:
fit_names = ["kG1p","kG1dp","kSa","kSi"]
param_inds1 = in(fit_names).(pnames)
p0 = log.(pvals[param_inds1])     # use log-tranformed parameter values to better sample parameter space during fitting


# -- Set bounds on parameter to use during fitting:
Δ_ulb1 = 2  # log10-width from initial param values to use for bounds of parameter fitting
ulb1 = zeros(length(p0),2)
for i in axes(ulb1,1)
    ulb1[i,:] = log.([10.0.^(log10(exp.(p0[i])) - Δ_ulb1), 10.0.^(log10(exp.(p0[i])) + Δ_ulb1)])
end


## Define loss/error function:
function loss(pvals_in, fitdata;
    param_inds=param_inds1,
    Co=Co,
    pvals0=pvals,
    D_inds=D_inds, k_inds=k_inds,
    R=R, dr=dr, sa=1/surfCF, vol=1/volCF,
    tf=tf, Nts=Nts, tol=1e-3
    )
    # -- Define useful params/constants:
    CoG1 = Co[3]    # total concentration of GAB1
    
    # -- Set parameter values defined using priors:
    x2 = eltype(pvals_in).(pvals0)      # convert initial parameter values to correct type (for ForwardDiff compatibility)
    x2[param_inds] = exp.(pvals_in)     # transform parameters back to correct scale
    
    # -- Compute solutions:
    sol, r_sol, t_sol = try
        pdesolver_fitting([x2; Co]; R=R, dr=dr, tf=tf, Nts=Nts, tol=tol, maxiters=20)
    catch
        return Inf
    end


    # -- Extract output of interest:
    PG1S_cyt = sol.PG1S[:,end] .+ sol.G2PG1S[:,end]   # cytoplasmic GAB1-SHP2
    PG1S_cyt_ave = NumericalIntegration.integrate(r_sol, PG1S_cyt.*r_sol.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_mem = sol.EG2PG1S[end] .* sa/vol     # membrane GAB1-SHP2, converted to volume units
    PG1Stot = PG1S_cyt_ave .+ PG1S_mem     # total GAB1-SHP2
    ŷ = PG1Stot[1]/CoG1*100.0    # model-predicted percent SHP2-bound GAB1
    

    # -- Calculate model loss/error:
    loss = (fitdata.μ .- ŷ).^2 ./ (fitdata.σ).^2
    if any(isnan.(loss))
        return Inf
    else
        return loss, ŷ, sol, r_sol, t_sol
    end
end



## Settings for optimization:
maxit1 = 30     # max iterations for first optimization
maxit2 = 30     # max iteration for second (final) optimization
f_min = 2e-3        # minimum value for terminating optimization early (prior to iterations/max time)
f_tol = 2e-3        # tolerance/accuracy compared to `f_min` for terminating optimization early (prior to iterations/max time)
fcall_lim = 1e8     # max number of function calls allowed
rngseed = 123

ad = Optimization.AutoForwardDiff()     # autodiff to use
f1 = OptimizationFunction((x,p)->loss(x,p)[1], ad)  # first opt func with larger dr
f = OptimizationFunction((x,p)->loss(x,p; dr=0.1)[1], ad)   # second opt func with dr=0.1
callback = function (p,l) # callback function to observe model fitting
    println("Loss: $l")
    return false # Tell it to not halt the optimization. If return true, then optimization stops
end




## Solve Optimization problem:
# -- Run optimization:
run_opt = false
save_fn = "fitted_parameters.csv"

if run_opt
    # -- Run first optimization with larger dr to speed up initial fitting:
    optprob1 = Optimization.OptimizationProblem(f1, p0, fitdata_df, lb = ulb1[:,1], ub = ulb1[:,2])
    optalg1 = MultistartOptimization.TikTak(101)        # optimization algorithm to use
    local_optalg1 = NLopt.LD_LBFGS()
    optsol1 = solve(optprob1, optalg1, local_optalg1, maxiters=maxit1, callback=callback)  # using LBFGS as local algorithm

    
    
    # -- Secondary and final optimization with the smaller (desired) dr=0.1:
    optprob = Optimization.OptimizationProblem(f, optsol1.u, fitdata_df, lb = ulb1[:,1], ub = ulb1[:,2])
    optalg = NLopt.LD_LBFGS()        # optimization algorithm to use
    optsol = solve(optprob, optalg, maxiters=maxit2, callback=callback)  # using LBFGS as local algorithm

    opt_pvals = exp.(optsol.u)
    opt_df = DataFrame(name=fit_names, value=opt_pvals)
    CSV.write(save_fn, opt_df)
end


## Load saved parameter fits from disk?
load_opt = true
if load_opt
    opt_df = CSV.read(save_fn, DataFrame)
    opt_pvals = opt_df.value
end

## Inspect model fit:
@time pred1 = loss(log.(opt_pvals), fitdata_df; dr=0.1)[2]   # get fitted model predictions
println("Percent SHP2-bound GAB1: $(pred1)")









## =============== Bayesian inference using Turing.jl =============== ##
# -- Update univariate parameter priors with MLE-fitted values (keep spread the same):
prior_pars_df2 = copy(prior_pars_df)
for i in eachindex(fit_names)
    expts_fitted_df = @subset(expts_df, occursin.(fit_names[i], :name))
    nameᵢ = expts_fitted_df.name[1]
    Pᵢ = [opt_pvals[i] expts_df.errs[1] expts_df.wts[1] expts_df.err_types[1]]
    μᵢ, σᵢ = log.(calcModeSpread(Pᵢ))   # calculate mean and std of lognormal distribution
    @subset!(prior_pars_df2, .!(:param.==nameᵢ))
    append!(prior_pars_df2, DataFrame(param=nameᵢ, μ=μᵢ, σ=σᵢ))
end
uvpars = @subset(prior_pars_df2, occursin.(r"kp|kdp|^D|kSa|kSi|kG1p|kG1dp",:param))


## Define Turing model:
@everywhere @model function turing_model(uv, mv, μ_data, σ_data, ::Type{P}=Vector{Float64};
    tf=tf,   # final simulation time
    dr=dr,
    Co=Co, EGF=EGF, 
    # r=r, t=t,
    vol=1/volCF, sa=1/surfCF,   # volume and surface area of cell
    rtol=1e-4) where {P}

    # -- Define useful params/constants:
    CoG1 = Co[3]    # total concentration of GAB1
    
    # -- Define parameter priors:
    μ_kG1p = uv[uv.param.=="kG1p",:].μ[1]
    μ_kG1dp = uv[uv.param.=="kG1dp",:].μ[1]
    μ_kSa = uv[uv.param.=="kSa",:].μ[1] 
    μ_kSi = uv[uv.param.=="kSi",:].μ[1]
    σ_kG1p = uv[uv.param.=="kG1p",:].σ[1]
    σ_kG1dp = uv[uv.param.=="kG1dp",:].σ[1]
    σ_kSa = uv[uv.param.=="kSa",:].σ[1]
    σ_kSi = uv[uv.param.=="kSi",:].σ[1]

    kG1p ~ truncated(LogNormal(μ_kG1p, σ_kG1p), upper=exp.(μ_kG1p)*1e3)
    kG1dp ~ truncated(LogNormal(μ_kG1dp, σ_kG1dp), upper=exp.(μ_kG1dp)*1e3)
    kSa ~ truncated(LogNormal(μ_kSa, σ_kSa), upper=exp.(μ_kSa)*1e3)
    kSi ~ truncated(LogNormal(μ_kSi, σ_kSi), upper=exp.(μ_kSi)*1e3)
    

    # -- Sample from priors for other parameters (that are not being inferred):
    kG2 = rand(mv.G2, 1)   # EGFR-GRB2 binding
    kG1 = rand(mv.G1, 1)   # GRB2-GAB1 binding
    kS2 = rand(mv.S2, 1)   # GAB1-SHP2 binding
    kEGF = rand(mv.EGF, 1)   # EGF-EGFR binding
    kdim = rand(mv.dim, 1)   # EGFR dimerization
    kp = rand(LogNormal(uv[uv.param.=="kp",:].μ[1], uv[uv.param.=="kp",:].σ[1]), 1)
    kdp = rand(LogNormal(uv[uv.param.=="kdp",:].μ[1], uv[uv.param.=="kdp",:].σ[1]), 1)
    Dsfk = rand(LogNormal(uv[uv.param.=="Dsfk",:].μ[1], uv[uv.param.=="Dsfk",:].σ[1]), 1)
    Dg2 = rand(LogNormal(uv[uv.param.=="Dg2",:].μ[1], uv[uv.param.=="Dg2",:].σ[1]), 1)
    Dg2g1 = rand(LogNormal(uv[uv.param.=="Dg2g1",:].μ[1], uv[uv.param.=="Dg2g1",:].σ[1]), 1)
    Dg2g1s2 = rand(LogNormal(uv[uv.param.=="Dg2g1s2",:].μ[1], uv[uv.param.=="Dg2g1s2",:].σ[1]), 1)
    Dg1 = rand(LogNormal(uv[uv.param.=="Dg1",:].μ[1], uv[uv.param.=="Dg1",:].σ[1]), 1)
    Dg1s2 = rand(LogNormal(uv[uv.param.=="Dg1s2",:].μ[1], uv[uv.param.=="Dg1s2",:].σ[1]), 1)
    Ds2 = rand(LogNormal(uv[uv.param.=="Ds2",:].μ[1], uv[uv.param.=="Ds2",:].σ[1]), 1)

    # -- Set parameter values defined using priors:
    Diffs = [Dsfk; Dg2; Dg2g1; Dg2g1s2; Dg1; Dg1s2; Ds2]
    kvals = [kS2[2]/kS2[1]; kS2[2]; kG1[1]; kG1[2]; kG2[2]/kG2[1]; kG2[2];
        kG1p; kG1dp; kSa; kSi; kp; kdp; kEGF[2]/kEGF[1]; kEGF[2]; EGF; kdim[2]/kdim[1]; kdim[2]]

    # -- Compute solutions:
    sol, r_sol = pdesolver_fitting([Diffs; kvals; Co]; R=R, dr=dr, tf=tf, Nts=Nts, tol=rtol, maxiters=20)


    # -- Extract output of interest:
    PG1S_cyt = sol.PG1S[:,end] .+ sol.G2PG1S[:,end]   # cytoplasmic GAB1-SHP2
    PG1S_cyt_ave = NumericalIntegration.integrate(r_sol, PG1S_cyt.*r_sol.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_mem = sol.EG2PG1S[end] .* sa/vol     # membrane GAB1-SHP2
    PG1Stot = PG1S_cyt_ave .+ PG1S_mem     # total GAB1-SHP2
    pct_bound = PG1Stot[1]/CoG1*100.0    # model-predicted percent SHP2-bound GAB1

    # -- Match model predictions to data:
    μ_data ~ truncated(LogNormal(log(pct_bound), σ_data), upper=100.0)
    return [Diffs; kvals]
end




## ============== Settings for Bayesian inference on ODE parameters using Turing.jl ============== ##
# -- Set AD backend:
Turing.setadbackend(:forwarddiff)

# -- Define number of chains, samples, and acceptance rate:
nsamps = 1000  # number of Monte Carlo samples to run (per chain)
nchains = 5     # number of chains to sample from
acc_rate = 0.65  # NUTS/HMCDA acceptance rate -- 0.65 is the recommended choice from the NUTS paper (https://doi.org/10.48550/arXiv.1111.4246)

# -- Choose MCMC sampler:
splr = NUTS(acc_rate)


# -- Instantiate Turing model:
model2use = turing_model(uvpars, mvpars, exp(fitdata_lognorm[1]), fitdata_lognorm[2])


# -- File names for saving results:
fn_chain = cwd*"/Turing results/Turing_res_$(nchains)-chains_$(nsamps)-spls.jld2"   # for saving MCMC chain(s)
fn_chain_cd = cwd*"/Turing results/Turing_res_$(nchains)-chains_$(nsamps)-spls_$(currentday)_$(currenttime).jld2"
fn_post = cwd*"/Turing results/Turing_res_$(nchains)-chains_$(nsamps)-spls_posteriors"   # for saving posteriors



## =============== Perform Bayesian inference using Turing =============== ##
# -- Run MCMC sampling:
run_sampling = false    # run Turing inference?
save_results = false    # save results to disk?
mcmc_parmeth = MCMCDistributed()

if run_sampling
    @time chain = Turing.sample(model2use, splr, mcmc_parmeth, nsamps, nchains; 
        init_params = repeat([opt_pvals], nchains),    # begin sampling around MAP estimates
        progress=true)
    
    # ===== Save parameter inference (chain) results ===== #
    if save_results
        jldsave(fn_chain, chain=chain)
        jldsave(fn_chain_cd, chain=chain)   # save with date information
    end

    # ===== Store parameter inference results in DataFrame format ===== #
    chain_df = DataFrame(chain)
    df_quant = DataFrame(quantile(chain))
elseif !run_sampling && isfile(fn_chain)
    chain = load_object(fn_chain)
    chain_df = CSV.File(fn_post*".csv") |> DataFrame
    df_quant = CSV.File(fn_post*"_quantiles.csv") |> DataFrame
end




## Save additional chain/quantile information:
using Suppressor    # for suppressing warnings when extracting generated_quantities
if @isdefined(chain)
    # -- Full chain information:
    nsamps = size(chain_df, 1)

    # -- Parameter means and quantiles:
    df_quant.mean .= mean(chain)[:, :mean]    # parameter means across all chains
    df_quant.param = df_quant.parameters

    if save_results
        # -- Save inference results to disk:
        CSV.write(fn_post*"_$(currentday)_$(currenttime).csv", chain_df)    # just the inferred parameters
        CSV.write(fn_post*".csv", chain_df) # just the inferred parameters
        CSV.write(fn_post*"_quantiles_$(currentday)_$(currenttime).csv", df_quant)
        CSV.write(fn_post*"_quantiles.csv", df_quant)
    end
end




## MCMC visualizations: For diagnosing convergence of Markov chains
display(summarystats(chain))
params = names(chain, :parameters)
chain_mapping =
    # mapping(params .=> "sample value") *
    mapping(params .=> log .=> "log(sample value)") *
    mapping(; color=:chain => nonnumeric, layout=dims(1) => renamer(params))
plt = data(chain) * mapping(:iteration) * chain_mapping * visual(Lines)
draw(plt, axis=(;width=200, height=100),
    facet=(linkxaxes=:none, linkyaxes=:none)
    ) |> display


## Plot fitted parameter posteriors: 
chain_df2 = @pipe chain_df |> 
    select(_, r"^k|fex") |>
    DataFrames.stack(_) |> 
    filter(:value => x -> !any(f -> f(x), (isinf, ismissing, isnothing, isnan)), _) |> 
    unique(_) |> 
    rename(_, :variable => :parameter) |> 
    groupby(_, :parameter) |> 
    @transform(_, :var=var(:value)) |> 
    sort(_, :var)

fig_posts = @pipe chain_df2 |> 
    (data(_) * (
        mapping(:parameter=>sorter(unique(_.parameter)), :value=>log10=>"log₁₀(value)", layout=:parameter=>sorter(unique(_.parameter))) * 
            visual(RainClouds)
        )
        ) |> 
    draw(_, 
        axis=(width=150, height=150),
        facet=(linkxaxes=:none, linkyaxes=:none),
    )
fig_posts |> display
save("images/Turing results/parameter_posteriors.png", fig_posts, px_per_unit=3)
save("images/Turing results/parameter_posteriors.pdf", fig_posts, pt_per_unit=1)




## ===== Prior and posterior predictive checks ===== ##
# -- Compute prior predictive checks:
ncheck = 500
prior_chain = sample(model2use, Prior(), ncheck)
missing_μ = similar([fitdata_lognorm[1]], Missing)
model_missing = turing_model(uvpars, mvpars, missing_μ[1], fitdata_lognorm[2])
prior_check = predict(model_missing, prior_chain);


## Compute posterior predictive checks: 
posterior_check = predict(model_missing, chain[rand(1:size(chain,1), ncheck÷nchains),:,:]);    # Note: takes a couple of minutes if running all chain samples
summarystats(posterior_check)



## Plot prior and posterior checks:
# -- Prior predictive check:
f_checks = Figure(); ax1 = Axis(f_checks[1,1], xlabel="log(% SHP2-bound GAB1)", ylabel="frequency", title="Prior predictive check")
@pipe prior_check.value[:,1,1] |> 
    filter(x->!isnan(x), _) |> 
    log.(_) |> 
    Makie.hist!(ax1, _)
vlines!(ax1, [log(100)])

# -- Posterior predictive check:
ax2 = Axis(f_checks[1,2], xlabel="log(% SHP2-bound GAB1)", ylabel="frequency", title="Posterior predictive check")
plt_check2 = @pipe posterior_check.value[:,1,:] |> 
    vcat(_...) |> 
    filter(x->.!isnan(x), _) |> 
    log.(_) |> 
    Makie.hist!(ax2, _)
plt_check3 = hist!(ax2, log.(rand(fitdata_ln_dist, size(posterior_check,1)*(size(posterior_check,3)))), transparency=true) # draw from fit data dist
leg_checks = Legend(f_checks[1,3],
    [plt_check2, plt_check3],
    ["Prediction\nprior/posterior", "Fit data"]
    )
display(f_checks)




## Plot posterior predictions and experimental data:
check_df = @pipe posterior_check.value[:,1,:] |> 
    vcat(_...) |> 
    filter(x->.!isnan(x), _) |>     # remove NaNs
    DataFrame(μ=mean(_), σ=std(_),
        type="Model"
        ) |> 
    vcat(_, @transform(fitdata_df, :type="Expt"))

# -- Plot comparison to exptl data:
@pipe check_df |> 
    data(_) * mapping(:type=>"", :μ=>"% SHP2-bound\nGAB1") *
        (mapping(color=:type)*visual(BarPlot) + mapping(:σ, :σ)*visual(Errorbars)) |> 
    draw(_, 
        axis=(width=100, height=120, limits=(nothing,nothing,0,nothing), 
            xgridvisible=false, ygridvisible=false)
        )








## Data retrodiction:
# -- Sample parameter posteriors:
chain_array = Array(chain_df[!, r"^k|^fex"])     # array of parameter posteriors

nretro = 2000    # number of posterior draws to make
indices = StatsBase.sample(1:size(chain_array,1), nretro, replace = false)


# -- Get samples from priors for uninferred parameters:
prior_array = zeros(nretro, length(pvals[.!param_inds1]))
for i in axes(prior_array, 1)
    # -- Uninferred parameters:
    kG2ᵢ = rand(mvpars.G2)   # EGFR-GRB2 binding
    kG1ᵢ = rand(mvpars.G1)   # GRB2-GAB1 binding
    kS2ᵢ = rand(mvpars.S2)   # GAB1-SHP2 binding
    kEGFᵢ = rand(mvpars.EGF)   # EGF-EGFR binding
    kdimᵢ = rand(mvpars.dim)   # EGFR dimerization
    kpᵢ = rand(LogNormal(uvpars[uvpars.param.=="kp",:].μ[1], uvpars[uvpars.param.=="kp",:].σ[1]))
    kdpᵢ = rand(LogNormal(uvpars[uvpars.param.=="kdp",:].μ[1], uvpars[uvpars.param.=="kdp",:].σ[1]))
    Dsfkᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dsfk",:].μ[1], uvpars[uvpars.param.=="Dsfk",:].σ[1]))
    Dg2ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dg2",:].μ[1], uvpars[uvpars.param.=="Dg2",:].σ[1]))
    Dg2g1ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dg2g1",:].μ[1], uvpars[uvpars.param.=="Dg2g1",:].σ[1]))
    Dg2g1s2ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dg2g1s2",:].μ[1], uvpars[uvpars.param.=="Dg2g1s2",:].σ[1]))
    Dg1ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dg1",:].μ[1], uvpars[uvpars.param.=="Dg1",:].σ[1]))
    Dg1s2ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Dg1s2",:].μ[1], uvpars[uvpars.param.=="Dg1s2",:].σ[1]))
    Ds2ᵢ = rand(LogNormal(uvpars[uvpars.param.=="Ds2",:].μ[1], uvpars[uvpars.param.=="Ds2",:].σ[1]))
    
    Diffsᵢ = [Dsfkᵢ; Dg2ᵢ; Dg2g1ᵢ; Dg2g1s2ᵢ; Dg1ᵢ; Dg1s2ᵢ; Ds2ᵢ]
    kvalsᵢ = [kS2ᵢ[2]/kS2ᵢ[1]; kS2ᵢ[2]; kG1ᵢ[1]; kG1ᵢ[2]; kG2ᵢ[2]/kG2ᵢ[1]; kG2ᵢ[2];
        kpᵢ; kdpᵢ; kEGFᵢ[2]/kEGFᵢ[1]; kEGFᵢ[2]; EGF; kdimᵢ[2]/kdimᵢ[1]; kdimᵢ[2]]
    prior_array[i,:] = [Diffsᵢ; kvalsᵢ]
end



## Perform retrodiction:
using ProgressMeter
@everywhere Random.seed!(123)

retro_df = DataFrame()      # initialize output DataFrame
retro_full_df = DataFrame() # for saving full spatiotemporal profiles
pvals_retro = DataFrame()   # for saving parameter values used for each retrodiction run

prog = Progress(nretro)     # initialize progress meter
lk = ReentrantLock()    # lock for multithreading

run_retro = false
Threads.@threads for j in eachindex(indices)
    ind = indices[j]
    # -- Set parameter values:
    pvalsⱼ = copy(pvals)
    # pvalsⱼ[param_inds2] = exp.(chain_array[j, :])
    pvalsⱼ[param_inds1] = chain_array[indices[j], :]
    pvalsⱼ[.!param_inds1] = prior_array[j, :]
    
    # -- Run model:
    solⱼ, r_solⱼ, t_solⱼ = pdesolver(Co, pvalsⱼ[D_inds], pvalsⱼ[k_inds]; R=R, dr=dr, tf=tf, Nts=Nts, tol=1e-4, maxiters=20)     # solve model
    
    # -- Extract outputs of interest:
    aSFKⱼ = solⱼ.aSFK
    PG1S_cytⱼ = solⱼ.PG1S .+ solⱼ.G2PG1S  # cytoplasmic GAB1-SHP2
    PG1S_cyt_aveⱼ = NumericalIntegration.integrate(r_solⱼ, PG1S_cytⱼ[:,end].*r_solⱼ.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_memⱼ = solⱼ.EG2PG1S[end] .* volCF/surfCF     # membrane GAB1-SHP2, converted to volume units
    PG1Stotⱼ = PG1S_cyt_aveⱼ .+ PG1S_memⱼ     # total GAB1-SHP2
    pct_PG1Sⱼ = PG1Stotⱼ[1]/CoG1*100.0
    
    # -- Save results:
    if isnan(pct_PG1Sⱼ) # if got NaNs, skip this draw
    else
        Threads.lock(lk) do
            # dfⱼ = DataFrame(t=[collect(0:tf/Nts:tf)], t_retro=[t_solⱼ], PG1S=[PG1S_cytⱼ], index=j)
            dfⱼ = DataFrame(pct_PG1S=pct_PG1Sⱼ, rss=abs2(fitdata_df.μ[1] - pct_PG1Sⱼ), r=r_solⱼ, aSFK_f=aSFKⱼ[:,end], G1S2_f=PG1S_cytⱼ[:,end], index=ind)
            df_fullⱼ = DataFrame(r=[r_solⱼ], t_sol=[t_solⱼ], sol=solⱼ, index=ind)
            append!(retro_df, dfⱼ)  # save model results
            append!(retro_full_df, df_fullⱼ)  # save full model results
            append!(pvals_retro, DataFrame(pvalsⱼ', pnames)) # save parameter values for current draw
        end
    end
    next!(prog) # update progress meter
end


## ===== Compute "summary" surfaces of interest ===== ##
r_med = retro_full_df.r[1]
# -- Extract surfaces for active SFKs and calculate "median" surface:
aSFK_stack = @pipe [retro_full_df.sol[i].aSFK for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].aSFK)..., :)) # reshape to 3D matrix

aSFK_med0 = median(aSFK_stack, dims=3)[:,:,1]   # "median" aSFK surface

# -- Interpolate values on dr=0.1 grid:
aSFK_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(aSFK_med,2)
    interp_i = linear_interpolation(r_med, aSFK_med0[:,i])
    aSFK_med[:,i] = interp_i(0:0.1:R)
end



## Extract surfaces for cytosolic GAB1-SHP2 and calculate median surface:
PG1S_stack = @pipe [(retro_full_df.sol[i].PG1S .+ retro_full_df.sol[i].G2PG1S) for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].PG1S)..., :)) # reshape to 3D matrix

PG1S_med0 = median(PG1S_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface

# -- Interpolate values on dr=0.1 grid:
PG1S_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(PG1S_med,2)
    interp_i = linear_interpolation(r_med, PG1S_med0[:,i])
    PG1S_med[:,i] = interp_i(0:0.1:R)
end


## Extract surfaces for cytosolic phospho-GAB1 and calculate median and upper/lower CI surfaces:
PG1_stack = @pipe [(retro_full_df.sol[i].PG1 .+ retro_full_df.sol[i].G2PG1) for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].PG1)..., :)) |>  # reshape to 3D matrix
    _ .+ PG1S_stack

PG1_med0 = median(PG1_stack, dims=3)[:,:,1]   # median pGAB1 surface
PG1_lo0 = [quantile(PG1_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1_stack,1)] # lower credible interval @ t=tf
PG1_up0 = [quantile(PG1_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1_stack,1)] # upper credible interval @ t=tf

# -- Interpolate median surface values on dr=0.1 grid:
PG1_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(PG1S_med,2)
    interp_i = linear_interpolation(r_med, PG1_med0[:,i])
    PG1_med[:,i] = interp_i(0:0.1:R)
end

# -- Interpolate upper/lower CIs @ t=tf:
pg1_inter_lo = linear_interpolation(r_med, PG1_lo0)
pg1_inter_up = linear_interpolation(r_med, PG1_up0)

PG1_lo = pg1_inter_lo(0:0.1:R)  # interpolate lower CI
PG1_up = pg1_inter_up(0:0.1:R)  # interpolate upper CI






## ===== Compare expt and model predictions for % SHP2-bound GAB1 ===== ##
set_theme!()
fig_fitcomp = @pipe retro_df |> 
    select(_, [:pct_PG1S, :index]) |>
    unique(_) |>  
    @combine(_, :μ=mean(:pct_PG1S), :σ=std(:pct_PG1S), 
        :σ_low=quantile(:pct_PG1S, 0.5-0.341),
        :σ_hi=quantile(:pct_PG1S, 0.5+0.341)
        ) |>
    @transform(_, :type="model",
        :σ_low=:μ-:σ_low,
        :σ_hi=:μ-:σ_hi
        ) |> 
    vcat(_, @transform(fitdata_df, :type="Expt", :σ_low=:σ, :σ_hi=:σ)) |> 
    data(_) * mapping(:type, :μ=>"% SHP2-bound\nGAB1") * 
        (mapping(color=:type)*visual(BarPlot) + mapping(:σ, :σ) * visual(Errorbars)) |> 
    draw(_, axis=(width=100, height=100, xgridvisible=false, ygridvisible=false,
        limits=(nothing,nothing,0,nothing))
        )
fig_fitcomp |> display
save("images/Turing results/retrodict-expt-comp_pct-SHP2-bound-GAB1.png", fig_fitcomp, px_per_unit=3)
save("images/Turing results/retrodict-expt-comp_pct-SHP2-bound-GAB1.pdf", fig_fitcomp, pt_per_unit=1)



## -- Format retrodicted data for looking at GAB1-SHP2/aSFK profiles:
retro4plt_df = @pipe retro_df |> 
    select(_, Not([:pct_PG1S, :rss])) |> 
    DataFrames.stack(_, Not([:r, :index])) |> 
    groupby(_, [:variable, :r]) |> 
    @combine(_, :median = median(:value), :uq=quantile(:value,0.5+0.341), :lq=quantile(:value,0.5-0.341))
    
best_df = @pipe @subset(retro_df, :rss.==minimum(:rss)) |>    # simulation with lowest fitting error
    select(_, Not([:pct_PG1S, :rss])) |> 
    DataFrames.stack(_, Not([:r, :index]))   


## Plot retrodicted model solutions for GAB1-SHP2 and aSFKs at t=5 min:
aog_plot = data(unique(select(retro4plt_df, [:r, :median, :variable]))) * 
        mapping(:r, :median, color=:variable, row=:variable) * visual(Lines) +
    data(retro4plt_df) * 
        mapping(:r, :lq, :uq, color=:variable, row=:variable) * visual(Band, alpha=0.25, transparency=true)

set_theme!()
fig_retro = draw(aog_plot,
    axis=(xlabel="r (µm)", ylabel="molec⋅µm⁻³", title="5 min EGF",
        limits=(0, R, 0, nothing), xgridvisible=false, ygridvisible=false,
        # limits=(0, R, nothing, nothing), xgridvisible=false, ygridvisible=false,
        height=100, width=110
        ),
    facet=(; linkxaxes=:none, linkyaxes=:none),
    )
fig_retro |> display

# -- Save plots of retrodicted data to disk:
save("images/Turing results/retrodiction_G1S2-profile_t=5min.png", fig_retro, px_per_unit=3)
save("images/Turing results/retrodiction_G1S2-profile_t=5min.pdf", fig_retro, pt_per_unit=1)








## ============== Plot ensemble-predicted surfaces of desired model outputs ============== ##
## Plot surfaces of aSFKs and GAB1-SHP2:
cmap = :turbo     # colormap for plots
lw = 1	# line width
δt = 5   # spacing between time points to plot
δr = 5   # spacing between spatial points to plot
r_plt = collect(0:0.1:R)    # spatial grid for plot
surfw = 200; surfh = 200
ax_surfs = (type=Axis3, width=surfw, height=surfh, #xreversed=true, yreversed=true,
    xgridvisible=true, ygridvisible=true, zgridvisible=true,
    xlabel="Time (min)", ylabel="r (μm)", title="5 min EGF")
update_theme!(fontsize=8, font="Arial")



## ===== Active SFKs ===== ##
aSFK_tf = @subset(retro4plt_df, occursin.("aSFK_f", :variable))

fig1, axis1, plt1 = mk.lines(repeat([tf], length(r_med)), aSFK_tf.r, aSFK_tf.lq,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis1, repeat([tf], length(r_med)), aSFK_tf.r, aSFK_tf.uq, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis1.limits=(0,tf, 0,R, 0,125)
axis1.zlabel="aSFK\n(molec·μm⁻³)"
Colorbar(fig1[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig1 |> display
save("images/aSFK_median-surf.png", fig1, px_per_unit=3)
save("images/aSFK_median-surf.pdf", fig1, pt_per_unit=1)



## ===== Total pGAB1 ===== ##
fig2, axis2, plt2 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis2, repeat([tf], length(tvec)), r_plt, PG1_up, 
    linewidth=lw, color=:red, linestyle=:dash)
mk.surface!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis2.limits=(0,tf, 0,R, 0,150)
axis2.zlabel="pGAB1\n(molec·μm⁻³)"
Colorbar(fig2[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig2 |> display
save("images/pGAB1_median-surf.png", fig2, px_per_unit=3)
save("images/pGAB1_median-surf.pdf", fig2, pt_per_unit=1)




## ===== Total pGAB1-SHP2 ===== ##
PG1S_tf = @subset(retro4plt_df, occursin.("G1S2",:variable))

fig3, axis3, plt3 = mk.lines(repeat([tf], length(r_med)), PG1S_tf.r, PG1S_tf.lq,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis3, repeat([tf], length(r_med)), PG1S_tf.r, PG1S_tf.uq, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], colormap=cmap, shading=false) 
mk.wireframe!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis3.limits = (0,tf, 0,R, 0,40) 
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display
save("images/pGAB1-SHP2_median-surf.png", fig3, px_per_unit=3)
save("images/pGAB1-SHP2_median-surf.pdf", fig3, pt_per_unit=1)