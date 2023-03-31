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


## Load necessary packages and solver functions:
using SharedArrays, CSV, DataFrames, DataFramesMeta, Pipe, Statistics, StatsBase
using AlgebraOfGraphics, CairoMakie
using Random, GlobalSensitivity, JLD2
using Base.Threads
using Distributed
Random.seed!(123)


## Using multithreading parallelism (local CPUs only):
# For VSCode, the number of threads available to Julia can be set in the Julia extension settings. In Juno (JuliaPro),
# the number of threads is automatically set based on the number of available cores. Check your 
# PC's environment variables to determine the number of available processors/threads on your system.
# For more information on setting computer cores, see https://docs.julialang.org/en/v1/manual/multi-threading/.
# Note that multithreading uses shared memory, so no need to call the @everywhere macro for the solver file.
Threads.nthreads()  # print the number of available threads


## Start workers on local machine using `Distributed` approach to parallelism:
use_distributed = false
if use_distributed
    if nprocs() == 1
        addprocs(Threads.nthreads(), exeflags="--project=.") # adds the same number of processes as available CPU cores
    end
    nprocs()    # check number of available processes
    nworkers()  # check number of available workers
end


## ===== Load necessary scripts and functions, define model parameters, etc. ===== ##
include("sapdesolver.jl") # Note: This defines a lot of variables used in this script. If it errors, try running again.



## ============== Define model parameters ============== ##
# Define the parameters that we wish to change, their names, and the names of the output variables:
par = Co # parameter values to change in GSA calculations 
par_names = ["[SFK]", "[GRB2]", "[GAB1]", "[SHP2]", "[EGFR]"]
par_names2 = ["[SFK]", "[GRB2]", "[GAB1]", "[SHP2]", "[EGFR]"]
var_names = ["r_1/2, SFK", "r_1/10, SFK", "r_1/2, pG1S2", "r_1/10, pG1S2", "[pG1S2]_cent:surf", "[pG1S2]_average"]

# Define allowable parameter ranges:
pbounds = []
lbm = 0.0002      # lower parameter bound multiplier
ubm = 2.0     # upper parameter bound multiplier
# lbm = 0.1      # lower parameter bound multiplier
# ubm = 10.0     # upper parameter bound multiplier
for i in eachindex(par)
    # temp_bounds = [par[i].*lbm, par[i].*ubm]
    temp_bounds = log.([par[i].*lbm, par[i].*ubm])
    push!(pbounds, temp_bounds)
end


## ============== eFAST GSA ============== ##
nruns_ef = 1000    # number of samples to take in the trajectories/design for eFAST → multiply this by # parameters for total number of samples
fn_efast = "eFAST-GSA-res_concs_$(nruns_ef)-spls-per-param"

run_efast = false
if run_efast
    # efast = gsa(fbatch_concs, eFAST(), pbounds; samples=nruns_ef, batch=true)
    efast = gsa(fbatch_concs_mt, eFAST(), pbounds; samples=nruns_ef, batch=true)
    S1 = efast.S1; S1[isnan.(S1)] .= 0.
    ST = efast.ST; ST[isnan.(ST)] .= 0.
    save(cwd*"/GSA results/$(fn_efast).jld2", "efast", efast)
    S1_df = DataFrame(efast.S1', var_names)
    ST_df = DataFrame(efast.ST', var_names)
    @transform!(S1_df, :param=par_names, :type="S1")
    @transform!(ST_df, :param=par_names, :type="ST")
    CSV.write(cwd*"/GSA results/$(fn_efast)_S1.csv", S1_df)
    CSV.write(cwd*"/GSA results/$(fn_efast)_ST.csv", ST_df)
end

## Load previously calculated eFAST GSA results, if available:
load_results = true
if ispath(cwd*"/GSA results/$(fn_efast)_S1.csv") && load_results
    efast = load(cwd*"/GSA results/$(fn_efast).jld2", "efast")
    S1_df = DataFrame(efast.S1', var_names)
    ST_df = DataFrame(efast.ST', var_names)
    @transform!(S1_df, :param=par_names, :type="S1")
    @transform!(ST_df, :param=par_names, :type="ST")
end

gsa_df = @pipe vcat(S1_df, ST_df) |> 
    DataFrames.stack(_, Not([:type, :param]))




## ===== Plot eFAST results ===== ##
update_theme!(fontsize=8, font="Arial")
cmap2 = cgrad(:batlow);

# -- Heatmaps:
efast_fig = @pipe gsa_df |> 
    @subset(_, :type.=="ST") |> 
    data(_) * mapping(:param=>sorter(par_names), :variable, :value, row=:type) *
    visual(Heatmap, colormap=cmap2) |> 
    draw(_, 
        axis=(;height=100, width=100, xticklabelrotation=pi/2, 
        title="Initial concentration eFAST analysis"
        )
    )
efast_fig |> display
save("images/$(fn_efast).png", efast_fig, px_per_unit=3)
save("images/$(fn_efast).pdf", efast_fig, pt_per_unit=1)
    
    
## Just the pGAB1-SHP2 outputs:
efast_g1s2_fig = @pipe gsa_df |> 
    @subset(_, occursin.("G1S2", :variable)) |>
    unstack(_, :type, :value) |> 
    @transform(_, :interactions=:ST.-:S1) |> 
    DataFrames.stack(_, Not([:variable, :param]), variable_name=:type) |> 
    # @subset(_, :type.=="ST") |> 
    @subset(_, :type.!="ST") |> 
    data(_) * mapping(:param=>sorter(par_names), :variable, :value, row=:type) *
        visual(Heatmap, colormap=cmap2) |> 
    draw(_, 
        axis=(;height=100, width=70, xticklabelrotation=pi/2, 
            title="Initial concentration eFAST analysis"
            ),
        colorbar=(spinewidth=1, width=16, height=80)
    )
efast_g1s2_fig |> display
save("images/$(fn_efast)_G1S2.png", efast_g1s2_fig, px_per_unit=3)
save("images/$(fn_efast)_G1S2.pdf", efast_g1s2_fig, pt_per_unit=1)