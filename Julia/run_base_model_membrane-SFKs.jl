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
	wongcols = aog.wongcolors();
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
include("get_param_posteriors.jl")	# may error on first run -- re-run if so


## Define parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]


## Define solver constants:
dr = 0.1 # length of each spatial step
R = 10.0 #  Radius of cell
Nts = 100 #  Number of time points to store
tf = 5.0 # final integration time
tvec = collect(0:tf/Nts:tf)

volCF = 1/(4/3*pi*R^3) # conversion factor from cell⁻¹ to umol/μm³ for a spherical cell
surfCF = 1/(4*pi*R^2) # conversion factor from cell⁻¹ to umol/μm² for a spherical cell

# -- Define initial species concentrations:
CoSFK = 6.0e5 * volCF  # Total concentration of SFK (molec/μm³)
CoG2 = 6.0e5 * volCF   # Total concentration of GRB2 (molec/μm³)
CoG1 = 6.0e5 * volCF   # Total concentration of GAB1 (molec/μm³)
CoS2 = 6.0e5 * volCF   # Total concentration of SHP2 (molec/μm³)
CoEGFR = 6.0e5 * surfCF    # Total concentration of EGFR (molec/μm²)
Co = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations
Cnames = ["CoSFK", "CoG2", "CoG1", "CoSHP2", "CoEGFR"]   # Names of initial concentration variables



## ========== Compare model predictions for cytosolic versus membrane-bound active SFKs ========== ##
# -- Generate parameter ensembles:
N = 2000		# number of parameter sets to use in ensembles
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)


## ===== Run ensemble simulations ===== ##
cyt_df = run_ensemble(pdesolver, ensemble, Co; R=R, Nts=Nts)	# cyto aSFK ensemble simulations (base model)
memb_df = run_ensemble(pdesolver_membSFK, ensemble, Co; R=R, Nts=Nts)	# membrane aSFK ensemble simulations



## ==== Get ensemble results: Baseline model with diffusible aSFKs ===== ##
# -- Get cytosolic GAB1-SHP2 predictions:
cyt_pg1s_tf = @pipe [cyt_df.sol[i].PG1Stot for i in axes(cyt_df,1)] |> 
	hcat(_...) |>   # horizontal concatenation
	reshape(_, (size(cyt_df.sol[1].PG1Stot)..., :))[:,end,:] # reshape to 3D matrix, get final time point only

cyt_pg1s_med = median(cyt_pg1s_tf, dims=2) |> vec	# GAB1-SHP2 median prediciton @ t=5 min for cytosolic aSFKs
cyt_pg1s_lo = [quantile(cyt_pg1s_tf[i,:], 0.5-0.341) for i in axes(cyt_pg1s_tf,1)] # lower credible interval @ t=tf
cyt_pg1s_hi = [quantile(cyt_pg1s_tf[i,:], 0.5+0.341) for i in axes(cyt_pg1s_tf,1)] # upper credible interval @ t=tf
	
	
	
	
## ===== Get ensemble results: SFKs active only at membrane ===== ##
r_med = memb_df.r[1]	# spatial grid from ensemble calculations

# -- Get cytosolic GAB1-SHP2 predictions:
PG1S_stack = @pipe [memb_df.sol[i].PG1Stot for i in axes(memb_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(memb_df.sol[1].PG1Stot)..., :)) # reshape to 3D matrix

PG1S_med0 = median(PG1S_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface
PG1S_lo0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1S_stack,1)] # lower credible interval @ t=tf
PG1S_up0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1S_stack,1)] # upper credible interval @ t=tf

# -- Interpolate values on dr=0.1 grid:
PG1S_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(PG1S_med,2)
    interp_i = linear_interpolation(r_med, PG1S_med0[:,i])
    PG1S_med[:,i] = interp_i(0:dr:R)
end

# -- Interpolate upper/lower CIs @ t=tf:
pg1s_inter_lo = linear_interpolation(r_med, PG1S_lo0)
pg1s_inter_up = linear_interpolation(r_med, PG1S_up0)

PG1S_lo = pg1s_inter_lo(0:0.1:R)  # interpolate lower CI
PG1S_up = pg1s_inter_up(0:0.1:R)  # interpolate upper CI




## ============== Plot desired outputs ============== ##
# -- Plot settings/parameters:
cmap = :turbo     # colormap for plots
lw = 1	# line width
δt = 5   # spacing between time points to plot
δr = 5   # spacing between spatial points to plot
r_plt = collect(0:0.1:R)    # spatial grid for plot
surfw = 200; surfh = 200
# surfw = 150; surfh = 150
ax_surfs = (type=Axis3, width=surfw, height=surfh, #xreversed=true, yreversed=true,
    xgridvisible=true, ygridvisible=true, zgridvisible=true,
    xlabel="Time (min)", ylabel="r (μm)", title="SFKs only active at plasma membrane")
# update_theme!(fontsize=12, font="Arial")
update_theme!(fontsize=8, font="Arial")


## Plot pGAB1-SHP2 surface:
# -- Credible interval projections @ t=tf:
fig3, axis3, plt3 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1S_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis3, repeat([tf], length(tvec)), r_plt, PG1S_up, 
    linewidth=lw, color=:red, linestyle=:dash)
	
# -- Projection @t=tf for median prediction w/ cytosolic aSFKs:
mk.lines!(axis3, repeat([tf], length(cyt_pg1s_med)), r_med, cyt_pg1s_med,
	linewidth=lw*2, color=wongcols[1])
mk.lines!(axis3, repeat([tf], length(cyt_pg1s_lo)), r_med, cyt_pg1s_lo,
	linewidth=lw, color=wongcols[1], linestyle=:dot)
mk.lines!(axis3, repeat([tf], length(cyt_pg1s_hi)), r_med, cyt_pg1s_hi,
	linewidth=lw, color=wongcols[1], linestyle=:dot)

# -- Surface & wire mesh:
mk.surface!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], colormap=cmap, shading=false) 
mk.wireframe!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis3.limits = (0,tf, 0,R, 0,40) 
# axis3.xticks=[0,5]; #axis3.zticks=0:1.5e5:4.5e5
# axis3.zlabel="pGAB1-SHP2 (cell⁻¹)"
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display
save("images/Membrane-aSFKS_pGAB1-SHP2_median-surf_N=$(N)-spls.png", fig3, px_per_unit=3)
save("images/Membrane-aSFKS_pGAB1-SHP2_median-surf_N=$(N)-spls.pdf", fig3, pt_per_unit=1)



## ===== Compare GAB1-SHP2 center-to-surface ratio b/w cytosolic versus membrane aSFKs ===== ##
# -- Center:surface ratio of GAB1-SHP2 @ tf:
csr_memb = DataFrame(value=PG1S_stack[1,end,:]./PG1S_stack[end,end,:], model="membrane")
csr_cyt = DataFrame(value=cyt_pg1s_tf[1,:]./cyt_pg1s_tf[end,:], model="cytosol")

## Compute Bayes factor for difference between distributions:
# -- Load RCall and R package for calculating Bayes factor:
using RCall
@rlibrary ggstatsplot	# for statistical visualization of comparison
@rlibrary BayesFactor	# for getting Bayes factors as a variable
@rlibrary easystats		# for interpreting Bayes factors -- using `effectsize` specifically
x = [csr_memb.model; csr_cyt.model]
y = [csr_memb.value; csr_cyt.value]

# -- Calculate Bayes factor for difference between distributions:
R"
ggstatsplot::ggbetweenstats(data.frame(x=$x, y=$y), x, y, 
	paired=TRUE,
	)
"

R"""
BFtest <- BayesFactor::ttestBF($(csr_memb.value), $(csr_cyt.value))	# compute Bayes factor
BF <- BayesFactor::extractBF(BFtest)	# extract Bayes factor
print(effectsize::interpret_bf(BF$bf))	# assess evidence for alternative hypothesis (diff. b/w means > 0)
"""

BF_res = rcopy(BFtest); print(BF_res)	# get and print Bayes factor results
println("ln(BF) = $(log.(rcopy(BF).bf)[1])")	# ln(BF)



## Plot the comparison:
# -- Barplot:
fig_csr_comp = @pipe vcat(csr_memb, csr_cyt) |> 
    groupby(_, :model) |> 
    @combine(_, :value=median(:value), :lci=quantile(:value, 0.5-0.341), :hci=quantile(:value, 0.5+0.341)) |> 
    # @combine(_, :value=median(:value), :lci=quantile(:value, 0.055), :hci=quantile(:value, 1-0.055)) |> 
    @transform(_, :lo=:value.-:lci, :hi=:hci.-:value) |> 
	data(_) * 
		mapping(:model=>"SFK activity", :value=>"Cg1s2(0)/Cg1s2(R)") * 
			(mapping(color=:model)*visual(BarPlot) + mapping(:lo, :hi)*visual(Errorbars)) |> 
    draw(_, axis=(width=70, height=80, limits=(nothing, (0., 1.05)), title="+EGF, 5 min"))
fig_csr_comp |> display
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_memb-SFKs.png", fig_csr_comp, px_per_unit=3)
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_memb-SFKs.pdf", fig_csr_comp, pt_per_unit=1)

# -- Raincloud plot:
fig_csr_comp2 = @pipe vcat(csr_memb, csr_cyt) |>  
    data(_) * 
        mapping(:model, :value=>"Cg1s2(0)/Cg1s2(R)", color=:model) * 
            visual(RainClouds, gap=0, violin_limits=(0,1)) |> 
    draw(_, axis=(;height=120, width=100, limits=(nothing, (0, 1.05)), title="+EGF, 5 min")) 
fig_csr_comp2 |> display
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_rainclouds_memb-SFKs.png", fig_csr_comp2, px_per_unit=3)
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_rainclouds_memb-SFKs.pdf", fig_csr_comp2, pt_per_unit=1)


## =============== Model predictions @ t=5 min for kinetic parameter sweeps =============== ##
# -- Get trace for base model with diffusible aSFKs:
sol_base, r_base = pdesolver(Co, Diffs, kvals)[1:2]
PG1S_trace = sol_base.PG1Stot[:,end]

## ===== Run parameter value sweep for parameter #1 ===== ##
# -- Sweep setup:
# sweep_k = "kG1dp"
sweep_k = "kS2r"
sweep_kind = occursin.(sweep_k, knames)
# sweep_vec = 10.0.^collect(-1:0.25:0)
# sweep_vec = [0.1, 0.3, 0.5, 0.7, 1.0]
sweep_vec = [0.01, 0.1, 0.25, 0.5, 1.0]
sweep_initk = kvals[sweep_kind][1]

# -- Sweep loop:
lk = ReentrantLock()
sweep_res = DataFrame()
Threads.@threads for i in eachindex(sweep_vec)
	pertᵢ = sweep_vec[i]	# parameter fold change i
	kvalsᵢ = copy(kvals)	# copy k values
	kvalsᵢ[sweep_kind] = kvals[sweep_kind].*pertᵢ	# perturb parameter value

	# -- Get model solution:
	solᵢ, rᵢ = pdesolver_membSFK(Co, Diffs, kvalsᵢ; R=R, dr=dr, tf=tf, Nts=Nts+1)[1:2]

	# -- Store the results:
	pg1sᵢ = solᵢ.PG1Stot[:,end]
	Threads.lock(lk) do
		dfᵢ = DataFrame(PG1S=pg1sᵢ, r=rᵢ, pert=pertᵢ, param=sweep_k)	
		append!(sweep_res, dfᵢ)
	end
end


## Plot sweep results:
swp_plth = 80
swp_pltw = swp_plth*1.2
# cmap2 = cgrad(:blues, 5, categorical = true)
cmap2 = cgrad(:greens, 5, categorical = true)

fig_sweep = @pipe sweep_res |> 
	groupby(_, :pert) |> 
	@transform(_, :PG1S = :PG1S./maximum(:PG1S)) |> 
	# @transform(_, :PG1S = :PG1S) |> 
	data(_) * # sweep curves
		mapping(:r, :PG1S, series=:pert=>string, color=:pert=>"$sweep_k\nfold-change") * 
			visual(Lines, linewidth=1.5, colormap=cmap2) +
	data(DataFrame(y=PG1S_trace./maximum(PG1S_trace), r=r_base)) * 	# base model prediction (diffusible SFKs)
	# data(DataFrame(y=PG1S_trace, r=r_sol₀)) * 	# base model prediction (diffusible SFKs)
		mapping(:r, :y) * 
			visual(Lines, color=:red, linestyle=:dash, linewidth=1.5)	|> 
	draw(_, axis=(width=swp_pltw, height=swp_plth, xtickwidth=1.5, ytickwidth=1.5, spinewidth=1.5,
		limits=(0, R, 0, 1), 
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="$(sweep_k) sweep",
		),
		colorbar=(ticks=LinRange(extrema(sweep_vec)..., 5), 
			tickwidth=0.8, spinewidth=0.8, width=swp_plth/8, height=swp_pltw/2
			)
	)
fig_sweep |> display
save("images/Membrane-aSFK_param-sweep_$(sweep_k).png", fig_sweep, px_per_unit=3)
save("images/Membrane-aSFK_param-sweep_$(sweep_k).pdf", fig_sweep, pt_per_unit=1)


@pipe sweep_res |> 
	groupby(_, :pert) |> 
	@transform(_, :PG1S = :PG1S./maximum(:PG1S)) |> 
	@subset(_, :r.==0.0) |> 
	sort(_, :pert) |> print
(PG1S_trace./maximum(PG1S_trace))[1]




## ===== Run parameter value sweep for parameter #2 ===== ##
# -- Sweep setup:
sweep_k2 = "kG1dp"
sweep_kind2 = occursin.(sweep_k2, knames)
# sweep_vec2 = 10.0.^collect(-1:0.25:0)
sweep_vec2 = [0.1, 0.3, 0.5, 0.7, 1.0]
# sweep_vec2 = [0.01, 0.1, 0.25, 0.5, 1.0]
sweep_initk2 = kvals[sweep_kind2][1]

# -- Sweep loop:
lk = ReentrantLock()
sweep_res2 = DataFrame()
Threads.@threads for i in eachindex(sweep_vec2)
	pertᵢ = sweep_vec2[i]	# parameter fold change i
	kvalsᵢ = copy(kvals)	# copy k values
	kvalsᵢ[sweep_kind2] = kvals[sweep_kind2].*pertᵢ	# perturb parameter value

	# -- Get model solution:
	solᵢ, rᵢ = pdesolver_membSFK(Co, Diffs, kvalsᵢ; R=R, dr=dr, tf=tf, Nts=Nts+1)[1:2]

	# -- Store the results:
	pg1sᵢ = solᵢ.PG1Stot[:,end]
	Threads.lock(lk) do
		dfᵢ = DataFrame(PG1S=pg1sᵢ, r=rᵢ, pert=pertᵢ, param=sweep_k2)	
		append!(sweep_res2, dfᵢ)
	end
end


## Plot sweep results:
swp_plth = 80
swp_pltw = swp_plth*1.2
cmap2 = cgrad(:blues, 5, categorical = true)

fig_sweep2 = @pipe sweep_res2 |> 
	groupby(_, :pert) |> 
	@transform(_, :PG1S = :PG1S./maximum(:PG1S)) |> 
	# @transform(_, :PG1S = :PG1S) |> 
	data(_) * # sweep curves
		mapping(:r, :PG1S, series=:pert=>string, color=:pert=>"$(sweep_k2)\nfold-change") * 
			visual(Lines, linewidth=1.5, colormap=cmap2) +
	data(DataFrame(y=PG1S_trace./maximum(PG1S_trace), r=r_base)) * 	# base model prediction (diffusible SFKs)
		mapping(:r, :y) * 
			visual(Lines, color=:red, linestyle=:dash, linewidth=1.5)	|> 
	draw(_, axis=(width=swp_pltw, height=swp_plth, xtickwidth=1.5, ytickwidth=1.5, spinewidth=1.5,
		limits=(0, R, 0, 1), 
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="$(sweep_k2) sweep",
		),
		colorbar=(ticks=LinRange(extrema(sweep_vec2)..., 5), 
			tickwidth=0.8, spinewidth=0.8, width=swp_plth/8, height=swp_pltw/2
			)
	)
fig_sweep2 |> display
save("images/Membrane-aSFK_param-sweep_$(sweep_k2).png", fig_sweep2, px_per_unit=3)
save("images/Membrane-aSFK_param-sweep_$(sweep_k2).pdf", fig_sweep2, pt_per_unit=1)


@pipe sweep_res2 |> 
	groupby(_, :pert) |> 
	@transform(_, :PG1S = :PG1S./maximum(:PG1S)) |> 
	@subset(_, :r.==0.0) |> 
	sort(_, :pert) |> println
(PG1S_trace./maximum(PG1S_trace))[1]




## =============== kG1dp perturbation =============== ##
# kG1dp_mult = 0.5
# kvals2 = [kS2f, kS2r, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp*kG1dp_mult, kSa, kSi, kp, kdp, kEGFf, kEGFr, EGF, kdf, kdr]
# @time sol3, r_sol3, t_sol3 = pdesolver_membSFK(Co, Diffs, kvals2; R=R, dr=dr, tf=tf, Nts=Nts+1) # Solve the model
# PG1Stot3 = sol3.PG1Stot #./volCF

# ## Plot pGAB1-SHP2 surface:
# x3 = t_sol3[1:δt:end]
# y3 = r_sol3[1:δr:end]
# z3 = PG1Stot3'[1:δt:end,1:δr:end]
# cmap3 = :turbo

# fig3, axis3, plt3 = mk.lines(repeat([tf], length(r_base)), r_base, PG1S_trace, color=:red, linestyle=:dash, axis=ax_surfs)
# mk.surface!(axis3, x3, y3, z3, colormap=cmap3, shading=false)#, axis=ax_surfs) 
# mk.wireframe!(axis3, x3, y3, z3, color=:black, linewidth=1)
# axis3.limits = (0, tf, 0, R, 0, nothing) 
# axis3.xticks = [0, tf]
# # axis3.zticks = zticks
# # axis3.zlabel = "pGAB1-SHP2 (cell⁻¹)"
# axis3.zlabel = "GAB1-SHP2\n(molec·μm⁻³)"
# axis3.title = "SFKs only active at plasma membrane; $(kG1dp_mult)x kG1dp"
# Colorbar(fig3[1,2], limits=(0,100), colormap=cmap3, spinewidth=theme_lw, 
# 	width=surfh/20, height=surfw/3, label="% max. concentration")
# fig3 |> display
# # save("images/Membrane-aSFK_$(kG1dp_mult)xkG1dp_pGAB1-SHP2.png", fig3,  px_per_unit=3)
# # save("images/Membrane-aSFK_$(kG1dp_mult)xkG1dp_pGAB1-SHP2.pdf", fig3,  pt_per_unit=1)



# ## =============== kS2r perturbation =============== ##
# kS2r_mult = 0.7
# kvals4 = [kS2f, kS2r*kS2r_mult, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp, kSa, kSi, kp, kdp, kEGFf, kEGFr, EGF, kdf, kdr]
# @time (sol4, r_sol4, t_sol4) = pdesolver_membSFK(Co, Diffs, kvals4; R=R, dr=dr, tf=tf, Nts=Nts+1) # Solve the model
# PG1Stot4 = sol4.PG1Stot #./volCF

# ## Plot pGAB1-SHP2 surface:
# x4 = t_sol4[1:δ:end]
# y4 = r_sol4[1:δ:end]
# z4 = PG1Stot4'[1:δ:end,1:δ:end]
# zticks4 = 0:20:120
# cmap4 = :turbo
# fig4, axis4, plt4 = mk.lines(repeat([tf], length(t_sol₀)), r_sol₀, PG1S_trace, color=lc, linestyle=ls, axis=ax_surfs)
# mk.surface!(axis4, x4, y4, z4, colormap=cmap4, shading=false)#, axis=ax_surfs) 
# mk.wireframe!(axis4, x4, y4, z4, color=:black, linewidth=1)
# axis4.limits = (0, tf, 0, R, 0, zticks4[end]) 
# axis4.xticks = [0,tf]
# axis4.zticks = zticks4
# # axis4.zlabel = "pGAB1-SHP2 (cell⁻¹)"
# axis4.zlabel = "GAB1-SHP2\n(molec·μm⁻³)"
# axis4.title = "SFKs only active at plasma membrane; $(kS2r_mult)x kS2r"
# Colorbar(fig4[1,2], limits=(0,100), colormap=cmap4, spinewidth=theme_lw, 
# 	width=surfh/20, height=surfw/3, label="% max. concentration")
# fig4 |> display
# # save("images/Membrane-aSFK_$(kS2r_mult)xkS2r_pGAB1-SHP2.png", fig4,  px_per_unit=3)
# # save("images/Membrane-aSFK_$(kS2r_mult)xkS2r_pGAB1-SHP2.pdf", fig4,  pt_per_unit=1)