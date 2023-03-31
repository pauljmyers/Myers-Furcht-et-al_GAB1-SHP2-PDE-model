## Necessary packages:
using Turing, CSV, DataFrames, JLD2, ProgressMeter, Distributed, Base.Threads
# using Interpolations, StatsBase

## Load priors:
include("get_param_priors.jl")

## Load "best-fit" parameter values for base model:
opt_df = CSV.read("fitted_parameters.csv", DataFrame)
opt_pvals = opt_df.value

## Load posterior distributions for fitted parameters:
chain = load_object("Turing results/Turing_res_5-chains_1000-spls.jld2")
chain_df = DataFrame(chain)

## Get highest-probability fitted parameter values:
best_pvals = @pipe log.(Array(select(chain_df, r"^k|^D"))) |> 
    median(_, dims=1) |> 
    vec(_) |> 
    exp.(_)


# -- Set baseline parameter values:
pnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2",
    "kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]

fit_inds = in(opt_df.name).(pnames)

pvals = copy(pvals0)
pvals[fit_inds] = best_pvals  # set highest-prob values for fitted parameters

Diffs = pvals[D_inds]
kvals = pvals[k_inds]


## Functions for generating and simulating parameter ensembles using posteriors and priors:
function generate_ensemble(chain, uvpars, mvpars, pnames; N=prod(size(chain)[[1,3]]))
    chain_df = DataFrame(chain)
    chain_array0 = Array(chain_df[!, r"^k|^D"])     # array of parameter posteriors
    fit_names = names(select(chain_df, r"^k|^D"))
    param_inds = in(fit_names).(pnames)
    Diff_inds = occursin.(r"^D", pnames)
    k_inds = occursin.(r"^k", pnames)
    
    indices = StatsBase.sample(1:size(chain_array0,1), N, replace = false)
    chain_array = chain_array0[indices, :]

    # -- Get samples from priors for uninferred parameters:
    pvals = zeros(N, length(pnames))
    Diffs = []
    kvals = []
    for i in axes(pvals, 1)
        # -- Uninferred parameters:
        kG2ᵢ = rand(mvpars.G2)   # EGFR-GRB2 binding
        kG1ᵢ = rand(mvpars.G1)   # GRB2-GAB1 binding
        kS2ᵢ = rand(mvpars.S2)   # GAB1-SHP2 binding
        kEGFᵢ = rand(mvpars.EGF)   # EGF-EGFR binding
        kdimᵢ = rand(mvpars.dim)   # EGFR dimerization
        kG1pᵢ = rand(LogNormal(uvpars[uvpars.param.=="kG1p",:].μ[1], uvpars[uvpars.param.=="kG1p",:].σ[1]))
        kG1dpᵢ = rand(LogNormal(uvpars[uvpars.param.=="kG1dp",:].μ[1], uvpars[uvpars.param.=="kG1dp",:].σ[1]))
        kSaᵢ = rand(LogNormal(uvpars[uvpars.param.=="kSa",:].μ[1], uvpars[uvpars.param.=="kSa",:].σ[1]))
        kSiᵢ = rand(LogNormal(uvpars[uvpars.param.=="kSi",:].μ[1], uvpars[uvpars.param.=="kSi",:].σ[1]))
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
            kG1pᵢ; kG1dpᵢ; kSaᵢ; kSiᵢ; kpᵢ; kdpᵢ; kEGFᵢ[2]/kEGFᵢ[1]; kEGFᵢ[2]; EGF; kdimᵢ[2]/kdimᵢ[1]; kdimᵢ[2]]

        pvals[i,:] = [Diffsᵢ; kvalsᵢ]
        # push!(Diffs, Diffsᵢ)
        # push!(kvals, kvalsᵢ)
    end
    pvals[:,param_inds] = chain_array
    Diffs = pvals[:, Diff_inds]
    kvals = pvals[:, k_inds]
    return pvals, Diffs, kvals
end
kS2f = kS2[2]/kS2[1]
kS2r = kS2[2]
kG2f = kG2[2]/kG2[1]
kG2r = kG2[2]
kG1f = kG1[1]
kG1r = kG1[2]
kEGFf = kEGF[2]/kEGF[1]
kEGFr = kEGF[2]
kdimf = kdf = kdim[2]/kdim[1]
kdimr = kdr = kdim[2]





## Ensemble simulation method(s) for the BASE model (no pulse-chase):
"""
    run_ensemble(model_fun, ensemble, Co;
        dr = 0.2,
        R = 10.0, 
        tf = 5.0, 
        Nts = 100, 
        tol = 1e-4,
        maxit = 20, 
        D_inds = 1:7, 
        k_inds = 8:24,
        show_prog = true
        )\n
Function for ensemble simulations with the GAB1-SHP2 reaction-diffusion model.\n
...
Function for quickly running model simulations with parameter ensembles. Returns a DataFrame 
containing the full model solutions for each parameter set. By default the simulations 
are multithreaded.

# Positional arguments
model_fun = PDE solver function to use\n
ensemble = matrix containing parameter sets to use for model ensemble simulations\n
Co = vector of initial species concentrations\n
# Keyword arguments
dr = spatial step size\n
R = simulation radius (radius of the cell)\n
tf = final simulation time\n
maxiters = the number of semi-implicit Euler iterations to compute before moving on\n
tol = error tolerance for semi-implicit finite difference scheme at cell membrane.\n
Diff_inds = indices for diffusivity values\n
k_inds = indices for kinetic parameter values\n
show_prog = Boolean --> whether to show progress meter for ensemble calculation
"""
function run_ensemble(model_fun, ensemble, Co; 
    dr=0.2, R=10.0, tf=5.0, Nts=100, tol=1e-4, maxit=20,
    D_inds=1:7, k_inds=8:24,
    show_prog=true,
    )
    retro_full_df = DataFrame() # for saving full model solutions

    if show_prog
        prog = Progress(size(ensemble,1))     # initialize progress meter
    end
    lk = ReentrantLock()    # lock for multithreading

    Threads.@threads for j in axes(ensemble,1)
    # for j in axes(ensemble,1)
        # -- Set parameter values:
        pvalsⱼ = ensemble[j,:]
        
        # -- Run model:
        solⱼ, r_solⱼ, t_solⱼ = model_fun(Co, pvalsⱼ[D_inds], pvalsⱼ[k_inds];
            R=R, dr=dr, tf=tf, Nts=Nts, tol=tol, maxiters=maxit)[1:3]
        if any(isnan.(solⱼ.PG1S))   # if NaNs detected/appear, skip this simulation
        else    # save the results otherwise
            # -- Save results:
            df_fullⱼ = DataFrame(r=[r_solⱼ], t_sol=[t_solⱼ], sol=solⱼ, index=j)
            Threads.lock(lk) do     # thread lock to prevent data-race condition
                append!(retro_full_df, df_fullⱼ)  # save results
            end
        end
        if show_prog
            next!(prog) # update progress meter
        end
    end
    return retro_full_df
end




## Ensemble methods for the pulse-chase solver:
"""
    run_ensemble_pc(model_fun, ensemble, Co; 
        dr = 0.2, 
        R = 10.0, 
        tf = 5.0, 
        Nts = 100, 
        tol = 1e-4,
        maxit = 20, 
        D_inds = 1:7, 
        k_inds = 8:24
        )\n
Function for ensemble calculations with the EGF-gefitinib pulse-chasesolver.\n
...
Function for quickly running model simulations with parameter ensembles. Returns a DataFrame 
containing the full model solutions for each parameter set. By default the simulations 
are multithreaded.

# Positional arguments
model_fun = PDE solver function to use\n
ensemble = matrix containing parameter sets to use for model ensemble simulations\n
Co = vector of initial species concentrations\n
# Keyword arguments
dr = spatial step size\n
R = simulation radius (radius of the cell)\n
tf = final simulation time\n
maxiters = the number of semi-implicit Euler iterations to compute before moving on\n
tol = error tolerance for semi-implicit finite difference scheme at cell membrane.\n
Diff_inds = indices for diffusivity values\n
k_inds = indices for kinetic parameter values
"""
function run_ensemble_pc(model_fun, ensemble, Co; 
    dr=0.2, R=10.0, 
    t_prechase=5.0,
    t_chase=2.0,
    Nts=100, tol=1e-4, maxit=20,
    D_inds=1:7, k_inds=8:24
    )
    retro_full_df = DataFrame() # for saving full model solutions

    prog = Progress(size(ensemble,1))     # initialize progress meter
    lk = ReentrantLock()    # lock for multithreading

    Threads.@threads for j in axes(ensemble,1)
        # for j in axes(ensemble,1)
        # -- Set parameter values:
        pvalsⱼ = ensemble[j,:]
        
        # -- Run model:
        solⱼ, r_solⱼ, t_solⱼ = model_fun(Co, pvalsⱼ[D_inds], pvalsⱼ[k_inds];
            R=R, dr=dr, t_prechase=t_prechase, t_chase=t_chase,
            Nts=Nts, tol=tol, maxiters=maxit)[1:3]
        if any(isnan.(solⱼ.PG1S))   # if NaNs detected/appear, skip this simulation
        else    # save the results otherwise
            # -- Save results:
            df_fullⱼ = DataFrame(r=[r_solⱼ], t_sol=[t_solⱼ], sol=solⱼ, index=j)
            Threads.lock(lk) do     # thread lock to prevent data-race condition
                append!(retro_full_df, df_fullⱼ)  # save results
            end
        end
        next!(prog) # update progress meter
    end
    return retro_full_df
end
