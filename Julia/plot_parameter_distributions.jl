## Activate project environment:
# Store current working directory where this script and the Project.toml and Manifest.toml
# files for this project are located:
cwd = pwd() # Get current directory

# Activate and precompile the project's environment:
using Pkg
Pkg.activate(cwd)
# Pkg.instantiate()     # installs all package dependencies for the current project/environment
# Pkg.precompile()    # may take a few minutes to complete


## Load Julia libraries:
using CSV, DataFrames, DataFramesMeta, Pipe
using CairoMakie, AlgebraOfGraphics
const mk = Makie
const aog = AlgebraOfGraphics

# -- Define (custom) default AlgebraOfGraphics theme:
if @isdefined(AlgebraOfGraphics) 
	theme_lw = 1.5	# theme linewidth
    # set_aog_theme!()
    update_theme!(
		fontsize=8,
		linewidth=theme_lw,
        # Axis=(
		# 	# topspinevisible=true, topspinecolor=:gray75,
		# 	# rightspinevisible=true, rightspinecolor=:gray75,
		# 	xtickwidth=theme_lw, ytickwidth=theme_lw, spinewidth=theme_lw, 
		# 	bottomspinecolor=:black, leftspinecolor=:black, xtickcolor=:black, ytickcolor=:black,
		# 	xticksize=3, yticksize=3
        # ),
        # Axis3=(
        #     xspinecolor_1=:darkgray,
        #     yspinecolor_1=:darkgray,
        #     zspinecolor_1=:darkgray,
        #     xspinecolor_2=:darkgray,
        #     yspinecolor_2=:darkgray,
        #     zspinecolor_2=:darkgray,
        #     xspinecolor_3=:darkgray,
        #     yspinecolor_3=:darkgray,
        #     zspinecolor_3=:darkgray
        #     )
    )
end 

## Get parameter distributions:
include("get_param_posteriors.jl")


## Define parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2,f", "kS2,r", "kG1,f", "kG1,r", "kG2,f", "kG2,r", "kG1p", "kG1dp",
    "kSa", "kSi", "kcatE", "kdp", "kE,f", "kE,r",  "EGF", "kdE,f", "kdE,r"]
pnames2 = [Dnames; knames]


## Generate distributions
N = 5000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
ensemble_df = DataFrame(ensemble, pnames)
CSV.write("parameter_ensemble.csv", ensemble_df)

## Plot posterior distributions for fitted parameters:
dist_plot = @pipe DataFrame(ensemble, pnames2) |> 
    DataFrames.stack(_) |> 
    @subset(_, .!occursin.(r"kG1p|kG1dp|kSa|kSi", :variable)) |> 
    @subset(_, :variable.!="EGF") |> 
    (data(_) * 
        mapping(:variable=>sorter(unique(_.variable)), :value=>log10=>"log₁₀(value)", layout=:variable=>sorter(unique(_.variable))) * 
            visual(RainClouds)
        ) |> 
    draw(_, 
        axis=(width=75, height=75, xgridvisible=false,),
        facet=(linkxaxes=:none, linkyaxes=:none),
    )
dist_plot |> display
save("images/non-fitted_parameter_distributions.png", dist_plot, px_per_unit=3)
save("images/non-fitted_parameter_distributions.pdf", dist_plot, pt_per_unit=1)


## Plot distributions for non-fitted parameters:
dist_plot = @pipe DataFrame(ensemble, pnames) |> 
    DataFrames.stack(_) |> 
    @subset(_, occursin.(r"kG1p|kG1dp|kSa|kSi", :variable)) |> 
    (data(_) * 
        mapping(:variable=>sorter(unique(_.variable)), :value=>log10=>"log₁₀(value)", layout=:variable=>sorter(unique(_.variable))) * 
            visual(RainClouds)
        ) |> 
    draw(_, 
        axis=(width=75, height=75, xgridvisible=false,),
        facet=(linkxaxes=:none, linkyaxes=:none),
    )
dist_plot |> display
save("images/posterior_distributions.png", dist_plot, px_per_unit=3)
save("images/posterior_distributions.pdf", dist_plot, pt_per_unit=1)