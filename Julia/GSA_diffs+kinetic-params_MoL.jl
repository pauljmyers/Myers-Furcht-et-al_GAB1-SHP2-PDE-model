#= Description:
This version of the global sensitivity analysis script is for performing LHS-based GSA using 
parallel processing capabilities and only using the diffusivity and kinetic rate constant parameters.
=#

# =============== Activate project environment =============== #
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

# -- Set number of threads to use for BLAS calculations:
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1) # if on Apple silicon Mac (M1, etc.), set this to 1 for best performance
using MKL


## Load necessary packages and solver functions:
using SharedArrays, CSV, DataFrames, DataFramesMeta, Pipe, Statistics, StatsBase
using AlgebraOfGraphics, CairoMakie
using Random, GlobalSensitivity, JLD2
using Base.Threads
using Distributed
Random.seed!(123)




## Start workers on local machine using `Distributed` approach to parallelism:
Threads.nthreads()  # print the number of available threads

use_distributed = false
if use_distributed
    if nprocs() == 1
        addprocs(Threads.nthreads(), exeflags="--project=.") # adds the same number of processes as available CPU cores
    end
    nprocs()    # check number of available processes
    nworkers()  # check number of available workers
end


## ===== Load necessary scripts and functions, define model parameters, etc. ===== ##  
#= 
Note: The file below defines a lot of variables used in this script. If it errors, try running again. It also loads the 
discretized model PDE system, which will take a few minutes to build and solve (even after subsequent calls).
=#
include("sapdesolver_MoL.jl") 



## ============== Define model parameters ============== ##
# -- Names of output variables:
var_names = ["r_1/2, SFK", "r_1/10, SFK", "r_1/2, pG1S2", "r_1/10, pG1S2", "[pG1S2]_cent:surf", "[pG1S2]_average"]

# -- Set baseline parameter values:
pvals_in = copy(pvals)


# -- Define allowable parameter ranges:
pbounds = []
ubm = 1000.0     # parameter bound multiplier
for i in eachindex(pvals_in)
    # temp_bounds = [pvals_in[i]./ubm, pvals_in[i].*ubm]
    temp_bounds = log.([pvals_in[i]./ubm, pvals_in[i].*ubm])
    # temp_bounds = log10.([pvals_in[i]./ubm, pvals_in[i].*ubm])
    push!(pbounds, temp_bounds)
end


## ============== eFAST GSA ============== ##
nruns_ef = 1000    # number of samples to take in the trajectories/design for eFAST → multiply this by # parameters for total number of samples
fn_efast = "eFAST-GSA-res_diffs+kinetic-params_$(nruns_ef)-spls-per-param"

run_efast = false
if run_efast
    # -- Perform eFAST:
    # @time efast = gsa(fbatch_dk, eFAST(), pbounds; samples=nruns_ef, batch=true)  # using distributed parallelism
    @time efast = gsa(fbatch_dk_mt, eFAST(), pbounds; samples=nruns_ef, batch=true) # using multithreading parallelism

    S1 = efast.S1; S1[isnan.(S1)] .= 0.
    ST = efast.ST; ST[isnan.(ST)] .= 0.

    # -- Save results:
    save(cwd*"/GSA results/$(fn_efast).jld2", "efast", efast)
    S1_df = DataFrame(efast.S1', var_names)
    ST_df = DataFrame(efast.ST', var_names)
    @transform!(S1_df, :param=pnames, :type="S1")
    @transform!(ST_df, :param=pnames, :type="ST")
    CSV.write(cwd*"/GSA results/$(fn_efast)_S1.csv", S1_df)
    CSV.write(cwd*"/GSA results/$(fn_efast)_ST.csv", ST_df)
end



## Load previously calculated eFAST GSA results, if available:
load_prev_res = true
if ispath(cwd*"/GSA results/$(fn_efast)_S1.csv") && load_prev_res
    efast = load(cwd*"/GSA results/$(fn_efast).jld2", "efast")
    S1_df = DataFrame(efast.S1', var_names)
    ST_df = DataFrame(efast.ST', var_names)
    @transform!(S1_df, :param=pnames, :type="S1")
    @transform!(ST_df, :param=pnames, :type="ST")
end


gsa_df = @pipe vcat(S1_df, ST_df) |> 
    DataFrames.stack(_, Not([:type, :param]))



## ===== Plot eFAST results ===== ##
update_theme!(fontsize=8, font="Arial")
cmap2 = cgrad(:batlow);

# -- Heatmaps:
efast_fig = @pipe gsa_df |> 
    # @subset(_, :type.=="ST") |> 
    data(_) * mapping(:param=>sorter(pnames), :variable, :value, row=:type) *
    visual(Heatmap, colormap=cmap2) |> 
    draw(_, 
    axis=(;height=100, width=300, xticklabelrotation=pi/2, 
    title="Diffusivity and kinetic parameter eFAST analysis"
    )
    )
efast_fig |> display
save("images/$(fn_efast).png", efast_fig, px_per_unit=3)
save("images/$(fn_efast).pdf", efast_fig, pt_per_unit=1)
    
    
## Just the pGAB1-SHP2 outputs:
efast_g1s2_fig = @pipe gsa_df |> 
    @subset(_, .!occursin.("1/10", :variable)) |> 
    @subset(_, occursin.("G1S2", :variable)) |>
    unstack(_, :type, :value) |> 
    @transform(_, :interactions=:ST.-:S1) |> 
    DataFrames.stack(_, Not([:variable, :param]), variable_name=:type) |> 
    # @subset(_, :type.=="ST") |> 
    @subset(_, :type.!="ST") |> 
    data(_) * mapping(:param=>sorter(pnames), :variable, :value, row=:type) *
        visual(Heatmap, colormap=cmap2) |> 
    draw(_, 
        axis=(;height=100, width=300, xticklabelrotation=pi/2, 
            title="Diffusivity and kinetic parameter eFAST analysis"
            ),
        colorbar=(spinewidth=1, width=16, height=80)
    )
efast_g1s2_fig |> display
save("images/$(fn_efast)_G1S2.png", efast_g1s2_fig, px_per_unit=3)
save("images/$(fn_efast)_G1S2.pdf", efast_g1s2_fig, pt_per_unit=1)