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
using CSV, NumericalIntegration, Interpolations
using DataFrames, DataFramesMeta, Pipe
using CairoMakie, AlgebraOfGraphics
const mk = Makie
const aog = AlgebraOfGraphics

# -- Define (custom) default AlgebraOfGraphics theme:
if @isdefined(AlgebraOfGraphics) 
	theme_lw = 1.5	# theme linewidth
    set_aog_theme!()
    update_theme!(
		fontsize=8,
		linewidth=theme_lw,
        Axis=(
			# topspinevisible=true, topspinecolor=:gray75,
			# rightspinevisible=true, rightspinecolor=:gray75,
			xtickwidth=theme_lw, ytickwidth=theme_lw, spinewidth=theme_lw, 
			bottomspinecolor=:black, leftspinecolor=:black, xtickcolor=:black, ytickcolor=:black,
			xticksize=3, yticksize=3
        ),
        Axis3=(
            xspinecolor_1=:darkgray,
            yspinecolor_1=:darkgray,
            zspinecolor_1=:darkgray,
            xspinecolor_2=:darkgray,
            yspinecolor_2=:darkgray,
            zspinecolor_2=:darkgray,
            xspinecolor_3=:darkgray,
            yspinecolor_3=:darkgray,
            zspinecolor_3=:darkgray
            )
    )
end 

## Include model solver function in current session:
include("basepdesolver.jl") # load finite-diff PDE solver
include("basepdesolver_rect.jl") # load finite-diff PDE solver for rectangular coordinates
include("get_param_posteriors.jl")  # if errors, try re-running


## Define parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]


## Define solver and spatiotemporal grid constants:
dr = 0.1 # length of each spatial step
R = 10.0 #  Radius of cell
Nts = 100 #  Number of time points to store
tf = 5.0 # final integration time
tvec = collect(0:tf/Nts:tf)

volCF = 1/(4/3*pi*R^3) # conversion factor from cell⁻¹ to umol/μm³ for a spherical cell
surfCF = 1/(4*pi*R^2) # conversion factor from cell⁻¹ to umol/μm² for a spherical cell

# -- Define initial species concentrations:
CoSFK = 6.0e5*volCF  # Total concentration of SFK (molec/μm³)
CoG2 = 6.0e5*volCF   # Total concentration of GRB2 (molec/μm³)
CoG1 = 6.0e5*volCF   # Total concentration of GAB1 (molec/μm³)
CoS2 = 6.0e5*volCF   # Total concentration of SHP2 (molec/μm³)
CoEGFR = 6.0e5*surfCF    # Total concentration of EGFR (molec/μm²)
Co = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations
Cnames = ["CoSFK", "CoG2", "CoG1", "CoSHP2", "CoEGFR"]   # Names of initial concentration variables


## Test PDE solver:
@time sol, r_sol, t_sol = pdesolver_rect(Co, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts)
@time sol_sph, r_sol_sph, t_sol_sph = pdesolver(Co, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts)


## Run ensemble calculations:
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
retro_sph_df = run_ensemble(pdesolver, ensemble, Co)
retro_rect_df = run_ensemble(pdesolver_rect, ensemble, Co)


## ===== Compute "summary" surfaces of interest ===== ##
r_med = retro_rect_df.r[1]	# spatial grid from ensemble calculations

## Extract surfaces for cytosolic GAB1-SHP2 and calculate median surface:
# -- Spherical results:
PG1S_sph_stack = @pipe [(retro_sph_df.sol[i].PG1S .+ retro_sph_df.sol[i].G2PG1S) for i in axes(retro_sph_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_sph_df.sol[1].PG1S)..., :)) # reshape to 3D matrix

# PG1S_sph_med0 = median(PG1S_sph_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface
# PG1S_sph_lo0 = [quantile(PG1S_sph_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1S_sph_stack,1)] # lower credible interval @ t=tf
# PG1S_sph_up0 = [quantile(PG1S_sph_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1S_sph_stack,1)] # upper credible interval @ t=tf

# # -- Interpolate values on dr=0.1 grid:
# PG1S_sph_med = zeros(length(0:0.1:R), Nts+1)
# for i in axes(PG1S_sph_med,2)
#     interp_i = linear_interpolation(r_med, PG1S_sph_med0[:,i])
#     PG1S_sph_med[:,i] = interp_i(0:0.1:R)
# end

# # -- Interpolate upper/lower CIs @ t=tf:
# PG1S_sph_inter_lo = linear_interpolation(r_med, PG1S_sph_lo0)
# PG1S_sph_inter_up = linear_interpolation(r_med, PG1S_sph_up0)

# PG1S_sph_lo = PG1S_sph_inter_lo(0:0.1:R)  # interpolate lower CI
# PG1S_sph_up = PG1S_sph_inter_up(0:0.1:R)  # interpolate upper CI



# -- Rectangular results:
PG1S_rect_stack = @pipe [(retro_rect_df.sol[i].PG1S .+ retro_rect_df.sol[i].G2PG1S) for i in axes(retro_rect_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_rect_df.sol[1].PG1S)..., :)) # reshape to 3D matrix

PG1S_rect_med0 = median(PG1S_rect_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface
PG1S_rect_lo0 = [quantile(PG1S_rect_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1S_rect_stack,1)] # lower credible interval @ t=tf
PG1S_rect_up0 = [quantile(PG1S_rect_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1S_rect_stack,1)] # upper credible interval @ t=tf

# -- Interpolate values on dr=0.1 grid:
PG1S_rect_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(PG1S_rect_med,2)
    interp_i = linear_interpolation(r_med, PG1S_rect_med0[:,i])
    PG1S_rect_med[:,i] = interp_i(0:0.1:R)
end

# -- Interpolate upper/lower CIs @ t=tf:
PG1S_rect_inter_lo = linear_interpolation(r_med, PG1S_rect_lo0)
PG1S_rect_inter_up = linear_interpolation(r_med, PG1S_rect_up0)

PG1S_rect_lo = PG1S_rect_inter_lo(0:0.1:R)  # interpolate lower CI
PG1S_rect_up = PG1S_rect_inter_up(0:0.1:R)  # interpolate upper CI



    


## ============== Plot ensemble-predicted surfaces of desired model outputs ============== ##
# -- Plot surfaces of aSFKs and GAB1-SHP2 for median parameter values:
cmap = :turbo     # colormap for plots
lw = 1	# line width
δt = 5   # spacing between time points to plot
δr = 5   # spacing between spatial points to plot
r_plt = collect(0:0.1:R)    # spatial grid for plot
surfw = 200; surfh = 200
# surfw = 150; surfh = 150
ax_surfs = (type=Axis3, width=surfw, height=surfh, #xreversed=true, yreversed=true,
    xgridvisible=true, ygridvisible=true, zgridvisible=true,
    xlabel="Time (min)", ylabel="r (μm)", title="5 min EGF")
# update_theme!(fontsize=12, font="Arial")
update_theme!(fontsize=8, font="Arial")






## ===== Total pGAB1-SHP2 ===== ##
fig3, axis3, plt3 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1S_rect_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis3, repeat([tf], length(tvec)), r_plt, PG1S_rect_up, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_rect_med'[1:δt:end,1:δr:end], colormap=cmap, shading=false) 
mk.wireframe!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_rect_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis3.limits = (0,tf, 0,R, 0,40) 
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
axis3.title="Rectangular coordinates"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls_rectangular-coord.png", fig3, px_per_unit=3)
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls_rectangular-coord.pdf", fig3, pt_per_unit=1)





## =============== Ensemble predictions at final time point =============== ##
## ===== Center-to-surface ratio of GAB1-SHP2 ===== ##
df_sph = DataFrame(ratio=PG1S_sph_stack[1,end,:]./PG1S_sph_stack[end,end,:], coord="spherical")
df_rect = DataFrame(ratio=PG1S_rect_stack[1,end,:]./PG1S_rect_stack[end,end,:], coord="rectangular")
fig_ratio = @pipe vcat(df_sph, df_rect) |> 
	data(_) * mapping(:coord=>sorter(["spherical","rectangular"]), :ratio, color=:coord)  * 
	visual(RainClouds, violin_limits=(0,1), gap=0) |> 
draw(_, 
	# axis=(ylabel="normalized\n% phosphorylated", title="5 min EGF",
	axis=(title="5 min EGF",
		xgridvisible=false, ygridvisible=false,
		limits=(nothing, (0,nothing)), 
		height=75, width=90)
	)
fig_ratio |> display
save("images/GAB1-SHP2_ensemble_center-surface-ratios_sphere-rect-comps_t=5min.png", fig_ratio, px_per_unit=3)
save("images/GAB1-SHP2_ensemble_center-surface-ratios_sphere-rect-comps_t=5min.pdf", fig_ratio, pt_per_unit=1)



## Compute Bayes factor for difference between distributions:
# -- Load RCall and R package for calculating Bayes factor:
using RCall
@rlibrary ggstatsplot
@rlibrary BayesFactor	# for getting Bayes factors as a variable
@rlibrary easystats		# for interpreting Bayes factors
x = [df_sph.coord; df_rect.coord]
y = [df_sph.ratio; df_rect.ratio]

# -- Calculate Bayes factor for difference between distributions:
R"
ggstatsplot::ggbetweenstats(data.frame(x=$x, y=$y), x, y, 
	paired=TRUE,
	)
"

R"""
BFtest <- BayesFactor::ttestBF($(df_sph.ratio), $(df_rect.ratio))	# compute Bayes factor
BF <- BayesFactor::extractBF(BFtest)	# extract Bayes factor
print(effectsize::interpret_bf(BF$bf))	# assess evidence for alternative hypothesis (diff. b/w means > 0)
"""

BF_res = @rget(BFtest); print(BF_res)	# get and print Bayes factor results
println("ln(BF) = $(log.(@rget(BF).bf)[1])")	# ln(BF)