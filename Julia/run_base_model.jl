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
include("get_param_posteriors.jl")


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
@time sol, r_sol, t_sol = pdesolver(Co, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts, tol=1e-2)




## ===== Run ensemble calculations ===== ##
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
retro_full_df = run_ensemble(pdesolver, ensemble, Co)







## ===== Compute "summary" surfaces of interest ===== ##
r_med = retro_full_df.r[1]	# spatial grid from ensemble calculations

# -- Extract surfaces for active SFKs and calculate "median" surface:
aSFK_stack = @pipe [retro_full_df.sol[i].aSFK for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].aSFK)..., :)) # reshape to 3D matrix

aSFK_med0 = median(aSFK_stack, dims=3)[:,:,1]   # "median" aSFK surface
aSFK_lo0 = [quantile(aSFK_stack[:,end,:][i,:], 0.5-0.341) for i in axes(aSFK_stack,1)] # lower credible interval @ t=tf
aSFK_up0 = [quantile(aSFK_stack[:,end,:][i,:], 0.5+0.341) for i in axes(aSFK_stack,1)] # upper credible interval @ t=tf

# -- Interpolate values on dr=0.1 grid:
aSFK_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(aSFK_med,2)
    interp_i = linear_interpolation(r_med, aSFK_med0[:,i])
    aSFK_med[:,i] = interp_i(0:0.1:R)
end

# -- Interpolate upper/lower CIs @ t=tf:
aSFK_inter_lo = linear_interpolation(r_med, aSFK_lo0)
aSFK_inter_up = linear_interpolation(r_med, aSFK_up0)

aSFK_lo = aSFK_inter_lo(0:0.1:R)  # interpolate lower CI
aSFK_up = aSFK_inter_up(0:0.1:R)  # interpolate upper CI



## Extract surfaces for cytosolic GAB1-SHP2 and calculate median surface:
PG1S_stack = @pipe [(retro_full_df.sol[i].PG1S .+ retro_full_df.sol[i].G2PG1S) for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].PG1S)..., :)) # reshape to 3D matrix

PG1S_med0 = median(PG1S_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface
PG1S_lo0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1S_stack,1)] # lower credible interval @ t=tf
PG1S_up0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1S_stack,1)] # upper credible interval @ t=tf

# -- Interpolate values on dr=0.1 grid:
PG1S_med = zeros(length(0:0.1:R), Nts+1)
for i in axes(PG1S_med,2)
    interp_i = linear_interpolation(r_med, PG1S_med0[:,i])
    PG1S_med[:,i] = interp_i(0:0.1:R)
end

# -- Interpolate upper/lower CIs @ t=tf:
pg1s_inter_lo = linear_interpolation(r_med, PG1S_lo0)
pg1s_inter_up = linear_interpolation(r_med, PG1S_up0)

PG1S_lo = pg1s_inter_lo(0:0.1:R)  # interpolate lower CI
PG1S_up = pg1s_inter_up(0:0.1:R)  # interpolate upper CI



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



## ===== Active SFKs ===== ##
fig1, axis1, plt1 = mk.lines(repeat([tf], length(tvec)), r_plt, aSFK_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis1, repeat([tf], length(tvec)), r_plt, aSFK_up, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis1.limits=(0,tf, 0,R, 0,150)
# axis1.xticks=[0,5]; #axis1.zticks=0:2e5:6e5
# axis1.zlabel="aSFK (cell⁻¹)"
axis1.zlabel="aSFK\n(molec·μm⁻³)"
Colorbar(fig1[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig1 |> display
save("images/aSFK_median-surf_N=$(N)-spls.png", fig1, px_per_unit=3)
save("images/aSFK_median-surf_N=$(N)-spls.pdf", fig1, pt_per_unit=1)




## ===== Total pGAB1 ===== ##
fig2, axis2, plt2 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis2, repeat([tf], length(tvec)), r_plt, PG1_up, 
    linewidth=lw, color=:red, linestyle=:dash)
mk.surface!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis2.limits=(0,tf, 0,R, 0,150)
# axis2.xticks=[0,5]; #axis2.zticks=0:2e5:6e5
# axis2.zlabel="pGAB1 (cell⁻¹)"
axis2.zlabel="pGAB1\n(molec·μm⁻³)"
Colorbar(fig2[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig2 |> display
save("images/pGAB1_median-surf_N=$(N)-spls.png", fig2, px_per_unit=3)
save("images/pGAB1_median-surf_N=$(N)-spls.pdf", fig2, pt_per_unit=1)




## ===== Total pGAB1-SHP2 ===== ##
fig3, axis3, plt3 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1S_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis3, repeat([tf], length(tvec)), r_plt, PG1S_up, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], colormap=cmap, shading=false) 
mk.wireframe!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis3.limits = (0,tf, 0,R, 0,40) 
# axis3.xticks=[0,5]; #axis3.zticks=0:1.5e5:4.5e5
# axis3.zlabel="pGAB1-SHP2 (cell⁻¹)"
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls.png", fig3, px_per_unit=3)
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls.pdf", fig3, pt_per_unit=1)




## ===== Compare ensemble predictions to fitting data ===== ##
# -- Exptl data for % SHP2-bound GAB1:
fitdata_df = CSV.read("exptl_pct_SHP2-bound-GAB1.csv", DataFrame)
fitdata_lognorm = log.(calcModeSpread([fitdata_df.μ[1] fitdata_df.σ[1] 1.0 0.0]))
fitdata_ln_dist = truncated(LogNormal(fitdata_lognorm[1], fitdata_lognorm[2]), upper=100.0)
fitdata_spls = DataFrame(pct_PG1S=rand(fitdata_ln_dist, 1000), type="expt")	# sample from lognormal dist for exptl data


# -- Calculate % SHP2-bound GAB1 for ensemble calculations:
pct_df = DataFrame()
lk = ReentrantLock()

Threads.@threads for j in axes(retro_full_df,1)
	solⱼ = retro_full_df.sol[j]
	r_solⱼ = retro_full_df.r[j]
	PG1S_cytⱼ = solⱼ.PG1S .+ solⱼ.G2PG1S  # cytoplasmic GAB1-SHP2
    PG1S_cyt_aveⱼ = NumericalIntegration.integrate(r_solⱼ, PG1S_cytⱼ[:,end].*r_solⱼ.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_memⱼ = solⱼ.EG2PG1S[end] .* volCF/surfCF     # membrane GAB1-SHP2, converted to volume units
    # @show PG1S_memⱼ
    PG1Stotⱼ = PG1S_cyt_aveⱼ .+ PG1S_memⱼ     # total GAB1-SHP2
    pct_PG1Sⱼ = PG1Stotⱼ[1]/CoG1*100.0

	pct_dfⱼ = DataFrame(pct_PG1S=pct_PG1Sⱼ, type="model")
	Threads.lock(lk) do
		append!(pct_df, pct_dfⱼ)
	end
end


## Plot model and expt comparison:
fig_fitcomp = @pipe pct_df |> 
	# vcat(_, fitdata_spls) |> 
    # data(_) * mapping(:type, :pct_PG1S=>"% SHP2-bound\nGAB1", color=:type)*
	# 	visual(RainClouds) |>
	groupby(_, :type) |> 
	@combine(_, :μ=median(:pct_PG1S), :σ=std(:pct_PG1S), 
        # :σ_low=quantile(:pct_PG1S, 0.5-0.341),	# ~68% credible interval (≈1σ)
        # :σ_hi=quantile(:pct_PG1S, 0.5+0.341)
        :σ_low=quantile(:pct_PG1S, 0.5-0.445),	# ~89% credible interval
        :σ_hi=quantile(:pct_PG1S, 0.5+0.445)
        ) |>
    @transform(_, :type="model",
        :σ_low=:μ-:σ_low,
        :σ_hi=:σ_hi-:μ
        ) |> 
    vcat(_, @transform(fitdata_df, :type="Expt", :σ_low=:σ, :σ_hi=:σ)) |> 
    data(_) * mapping(:type, :μ=>"% SHP2-bound\nGAB1") * 
        # (mapping(color=:type)*visual(BarPlot) + mapping(:σ, :σ) * visual(Errorbars)) |>
        (mapping(color=:type)*visual(BarPlot) + mapping(:σ_low, :σ_hi) * visual(Errorbars)) |>
    draw(_, axis=(width=75, height=75, xgridvisible=false, ygridvisible=false,
        limits=(nothing,nothing,0,nothing))
        )
fig_fitcomp |> display
save("images/expt-model-comp_pct-SHP2-bound-GAB1.png", fig_fitcomp, px_per_unit=3)
save("images/expt-model-comp_pct-SHP2-bound-GAB1.pdf", fig_fitcomp, pt_per_unit=1)







## =============== Ensemble predictions at final time point =============== ##
# ===== aSFK and pGAB1-SHP2 predictions ===== #
fig_tf = @pipe DataFrame(r=repeat(r_plt,2),
	med=[PG1S_med[:,end]; aSFK_med[:,end]], 
	lo=[PG1S_lo; aSFK_lo], 
	up=[PG1S_up; aSFK_up], 
	variable=[repeat(["PG1S"], length(PG1S_lo)); repeat(["aSFK"], length(aSFK_lo))]
	) |> 
	# @transform(_, :lo=:med.-:lo, :up=:up.-:med) |> 
	data(_) *
		(mapping(:r, :med, color=:variable, row=:variable) * visual(Lines) +
		mapping(:r, :lo, :up, color=:variable, row=:variable) * visual(Band, alpha=0.25, transparency=true)) |> 
	draw(_,
		axis=(xlabel="r (µm)", ylabel="molec⋅µm⁻³", title="5 min EGF",
			xgridvisible=false, ygridvisible=false,
			limits=(0, R, 0, nothing), 
			height=75, width=90
			),
		facet=(; linkxaxes=:none, linkyaxes=:none),
		)
fig_tf |> display
save("images/retrodiction_G1S2-aSFK-profiles_t=5min.png", fig_tf, px_per_unit=3)
save("images/retrodiction_G1S2-aSFK-profiles_t=5min.pdf", fig_tf, pt_per_unit=1)



## ===== Center-to-surface ratio of GAB1-SHP2 ===== ##
fig_ratio = @pipe DataFrame(ratio=PG1S_stack[1,end,:]./PG1S_stack[end,end,:], x="GAB1-SHP2") |> 
	data(_) * mapping(:x, :ratio)  * 
	visual(RainClouds, color=aog.wongcolors()[1]) |> 
draw(_, 
	# axis=(ylabel="normalized\n% phosphorylated", title="5 min EGF",
	axis=(title="5 min EGF",
		xgridvisible=false, ygridvisible=false,
		limits=(nothing, (0,nothing)), 
		height=75, width=90)
	)
fig_ratio |> display
save("images/GAB1-SHP2_ensemble_center-surface-ratios_t=5min.png", fig_ratio, px_per_unit=3)
save("images/GAB1-SHP2_ensemble_center-surface-ratios_t=5min.pdf", fig_ratio, pt_per_unit=1)



## ===== Percentages of pEGFR and pGAB1 (amplification plot) ===== ##
# -- Compute percent phospho-GAB1:
PG1_cyt_ave = [NumericalIntegration.integrate(r_med, PG1_stack[:,end,i].*r_med.^2).*3.0./R^3 for i in axes(PG1_stack,3)]
pct_PG1 = ([(retro_full_df.sol[i].EG2PG1 .+ retro_full_df.sol[i].EG2PG1S)[end] for i in axes(retro_full_df,1)]/surfCF*volCF .+ PG1_cyt_ave) ./CoG1 .* 100.

# -- Compute percent phospho-EGFR:
pct_pE = @pipe retro_full_df |> 
	[(_.sol[i].E .+ _.sol[i].EG2 .+ _.sol[i].EG2G1 .+ _.sol[i].EG2PG1 .+ _.sol[i].EG2PG1S) for i in axes(retro_full_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_full_df.sol[1].E)..., :))[end,:] |>  	# reshape to 2D matrix
	_./CoEGFR.*100.0	# calculate % phospho-EGFR


## Plot results:
fig_amp = @pipe DataFrame(value=[pct_PG1; pct_pE], 
	variable=[repeat(["GAB1"], length(pct_PG1)); repeat(["EGFR"], length(pct_pE))],
	id=[collect(1:length(pct_PG1)); collect(1:length(pct_PG1))]
	) |>
	# groupby(_, :id) |>  
	unstack(_, :variable, :value) |> 
	@transform(_, :GAB1=:GAB1./:EGFR, :EGFR=:EGFR./:EGFR) |> 
	DataFrames.stack(_, Not(:id)) |> 
	groupby(_, :variable) |> 
	# @combine(_, :value = median(:value), lci=quantile(:value, 0.5-0.341), uci=quantile(:value, 0.5+0.341)) |> 
	# @transform(_, :lo=:value .- :lci, :hi=:uci .- :value) |> 
	# data(_) * mapping(:variable, :value) *
	# 	(mapping(color=:variable)*visual(BarPlot) + mapping(:lo, :hi)*visual(Errorbars)) |> 
	@subset(_, :variable.=="GAB1") |> 
	data(_) * mapping(:variable, :value=>log10=>"log₁₀(pGAB1/pEGFR)") * 
		visual(RainClouds, color=aog.wongcolors()[3]) |> 
	draw(_, 
		# axis=(ylabel="normalized\n% phosphorylated", title="5 min EGF",
		axis=(title="5 min EGF",
			xgridvisible=false, ygridvisible=false,
			limits=(nothing, (0,nothing)), 
			height=75, width=90)
		)
fig_amp |> display
save("images/amplification_pEGFR-pGAB1_t=5min.png", fig_amp, px_per_unit=3)
save("images/amplification_pEGFR-pGAB1_t=5min.pdf", fig_amp, pt_per_unit=1)




## ===== Plots of EGFR-bound GAB1-SHP2 fraction ===== ##
PG1S_cyt_ave = [NumericalIntegration.integrate(r_med, PG1S_stack[:,end,i].*r_med.^2).*3.0./R^3 for i in axes(PG1S_stack,3)]

s2_data = @pipe retro_full_df |> 	# AoG plot data
	[_.sol[i].EG2PG1S[end]/(_.sol[i].EG2PG1S[end] + PG1S_cyt_ave[i]*surfCF/volCF) for i in axes(retro_full_df,1)] |> 
	vcat(_...) |> 
	DataFrame(value=_, variable="memb_G1S2") |> 
	data(_)

# -- Raincloud plot in log-scale:
s2_fig1 = @pipe s2_data * 
	mapping(:variable, :value=>log10=>"log₁₀(frac EGFR-bound GAB1-SHP2)") * visual(RainClouds) |> 
	draw(_, 
		axis=(width=100, height=100,) #limits=(nothing, (0,1)))
		)
s2_fig1 |> display
save("images/pct-EGFR-bound-G1S2_log10_t=5min.png", s2_fig1, px_per_unit=3)
save("images/pct-EGFR-bound-G1S2_log10_t=5min.pdf", s2_fig1, pt_per_unit=1)


# -- Raincloud plot:
s2_fig2 = @pipe s2_data * 
		mapping(:variable, :value=>"frac. EGFR-bound GAB1-SHP2") * visual(RainClouds) |> 
	draw(_, 
		axis=(width=100, height=100,) #limits=(nothing, (0,1)))
		)
s2_fig2 |> display
save("images/pct-EGFR-bound-G1S2_t=5min.png", s2_fig2, px_per_unit=3)
save("images/pct-EGFR-bound-G1S2_t=5min.pdf", s2_fig2, pt_per_unit=1)


# -- Barplot:
s2_fig3 = @pipe retro_full_df |> 
	[_.sol[i].EG2PG1S[end]/(_.sol[i].EG2PG1S[end] + PG1S_cyt_ave[i]*surfCF/volCF) for i in axes(retro_full_df,1)] |> 
	vcat(_...) |> 
	DataFrame(value=_, variable="memb_G1S2") |> 
	groupby(_, :variable) |> 
	@combine(_, :μ=median(:value), :lo=quantile(:value, 0.5-0.341), :hi=quantile(:value, 0.5+0.341)) |> 
	@transform(_, :lo=:μ.-:lo, :hi=:hi.-:μ) |> 
	data(_) * mapping(:variable, :μ=>"frac. EGFR-bound GAB1-SHP2") * 
		(visual(BarPlot, color=aog.wongcolors()[1]) + mapping(:lo, :hi)*visual(Errorbars, color=:black)) |> 
	draw(_, 
		axis=(width=50, height=100, limits=(nothing, (0,nothing)))
		)
s2_fig3 |> display
save("images/pct-EGFR-bound-G1S2_barplot_t=5min.png", s2_fig3, px_per_unit=3)
save("images/pct-EGFR-bound-G1S2_barplot_t=5min.pdf", s2_fig3, pt_per_unit=1)












## =============== Steady-state calculations with parameter perturbations =============== ##
#=
The following code is for obtaining model results at the final time tf when perturbing a
kinetic rate constant by a factor of 10 above and below its baseline value and when also 
optionally perturbing the initial concentration of a protein species in the model.
=#
# -- Plot size settings:
pert_plth = 75
pert_pltw = pert_plth*1.2


## =============== Run simulations when changing diffusivities ============== ##
# -- Diffusivity to change
pert_D = r"Dsfk$"        # regex for rate constant of interest
pert_Dind = match.(pert_D, Dnames) .!= nothing  # get which entry in name vector matches desired parameter name
pert_Dname = Dnames[pert_Dind]
# pert_vecD = [0.01, 1., 100.]      # vector of multipliers for parameter perturbations
pert_vecD = [0.01, 1.]      # vector of multipliers for parameter perturbations

pertD_df = DataFrame()
lk = ReentrantLock()
Threads.@threads for i in eachindex(pert_vecD)
	# Generate solution for perturbation i:
	Diffsᵢ = copy(Diffs)
	Diffsᵢ[pert_Dind] = Diffsᵢ[pert_Dind] * pert_vecD[i]       # perturb kinetic rate constant
	(ss, r_ss) = pdesolver(Co, Diffsᵢ, kvals; R=R, dr=dr, tf=tf, Nts=Nts) # Solve the model
	PG1S_ss = ss.PG1Stot[:,end]./maximum(ss.PG1Stot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	PG1_ss = ss.PG1tot[:,end]./maximum(ss.PG1tot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	dfᵢ = DataFrame(PG1S=PG1S_ss, PG1=PG1_ss, r=r_ss, param=pert_D, pert=replace(string(pert_vecD[i]), r".0$"=>"").*"-fold")
	Threads.lock(lk) do 
		append!(pertD_df, dfᵢ)
	end
end

##
# -- Plot:
# ls_ord = ["1-fold","0.01-fold","100-fold"]
ls_ord = ["1-fold","0.01-fold"]

fig_pert_D = @pipe pertD_df |> 
    data(_) * mapping(:r, :PG1S, color=:pert=>sorter(ls_ord), linestyle=:pert=>sorter(ls_ord)) * visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth, limits=(0,R,0,1),
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="$(pert_Dname[1]) sensitivity",
		)
	)
fig_pert_D |> display
save("images/steady-state_PG1S_pert-$(pert_Dname[1]).pdf", fig_pert_D, pt_per_unit=1)







## =============== Run simulations when changing kinetic parameters ============== ##
# Kinetic parameter to change:
# pert_k = r"kG1dp"        # regex for rate constant of interest
# pert_k = r"kSi"        # regex for rate constant of interest
pert_k = r"kS2r"        # regex for rate constant of interest
# pert_k = r"kS2f"        # regex for rate constant of interest
pert_kind = match.(pert_k, knames) .!= nothing  # get which entry in name vector matches desired parameter name
pert_kname = knames[pert_kind][1]
# pert_vec = [0.1, 1., 10.]      # vector of multipliers for parameter perturbations
pert_vec = [0.01, 1., 100.]      # vector of multipliers for parameter perturbations

pert_df = DataFrame()
Threads.@threads for i in eachindex(pert_vec)
	# -- Generate solution for perturbation i:
	kvalsᵢ = copy(kvals)
	kvalsᵢ[pert_kind] = kvalsᵢ[pert_kind] * pert_vec[i]       # perturb kinetic rate constant
	(ss, r_ss) = pdesolver(Co, Diffs, kvalsᵢ; R=R, dr=dr, tf=tf, Nts=Nts) # Solve the model
	PG1S_ss = ss.PG1Stot[:,end]./maximum(ss.PG1Stot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	PG1_ss = ss.PG1tot[:,end]./maximum(ss.PG1tot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	dfᵢ = DataFrame(PG1S=PG1S_ss, PG1=PG1_ss, r=r_ss, param=pert_k, 
		pert=replace(string(pert_vec[i]), r".0$"=>"").*"-fold"
		)
	Threads.lock(lk) do
		append!(pert_df, dfᵢ)
	end
end

# -- Plot:
# ls_ord = ["1-fold","0.1-fold","10-fold"]
ls_ord = ["1-fold","0.01-fold","100-fold"]
# ls_ord = ["1-fold","0.001-fold","1000-fold"]
fig_pert_k = @pipe pert_df |> 
    data(_) * mapping(:r, :PG1S, color=:pert=>sorter(ls_ord), linestyle=:pert=>sorter(ls_ord)) * visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth, limits=(0,R, 0,1),
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="$(pert_k.pattern) sensitivity",
		)
	)
fig_pert_k |> display
save("images/steady-state_PG1S_pert-$(pert_kname).pdf", fig_pert_k, pt_per_unit=1)





## =============== Run simulations when changing concentrations and kinetic parameters ============== ##
# -- Kinetic parameter to change:
# pert_k = [r"kSi", r"kG1dp"]        # regex for rate constant of interest
pert_k = ["kSi", "kG1dp"]        # regex for rate constant of interest
pert_knames = join(pert_k,"-")
pert_vec = [1., 100.]      # vector of multipliers for parameter perturbations

# -- Initial species concentration to change:
# pert_C = r"SFK"       # regex for initial protein concentration of interest
pert_C = r"SHP2"       # regex for initial protein concentration of interest
pert_Cind = match.(pert_C, Cnames) .!= nothing
pert_Cname = Cnames[pert_Cind][1]
pert_vecC = [1., 10.]
Co2 = copy(Co); 
Co2[pert_Cind] = Co2[pert_Cind]#*pert_vecC


pert_kC_df = DataFrame()
for j in eachindex(pert_vecC)
	Coⱼ = copy(Co)
    Coⱼ[pert_Cind] = Coⱼ[pert_Cind] * pert_vecC[j]
	for k in eachindex(pert_k)
		pert_kindₖ = occursin.(pert_k[k], knames)  # get which entry in name vector matches desired parameter name
		pert_knameₖ = knames[pert_kindₖ][1]
		Threads.@threads for i in eachindex(pert_vec)
			# Generate solution for perturbation i:
			kvalsᵢ = copy(kvals) 
			kvalsᵢ[pert_kindₖ] = kvalsᵢ[pert_kindₖ] * pert_vec[i]       # perturb kinetic rate constant
			@time ss, r_ss = pdesolver(Coⱼ, Diffs, kvalsᵢ; R=R, dr=dr, tf=tf, Nts=Nts) # Solve the model
			PG1S_ss = ss.PG1Stot[:,end]./maximum(ss.PG1Stot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
			PG1_ss = ss.PG1tot[:,end]./maximum(ss.PG1tot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
			dfᵢ = DataFrame(PG1S=PG1S_ss, PG1=PG1_ss, r=r_ss, 
				param=pert_knameₖ, species=pert_C.pattern,
				pertk=replace(string(pert_vec[i]), r".0$"=>"").*"x",
				pertC=replace(string(pert_vecC[j]), r".0$"=>"").*"x",
				)
			Threads.lock(lk) do
				append!(pert_kC_df, dfᵢ)
			end
		end
	end
end
@transform!(pert_kC_df, @byrow :condition = "$(:pertk) $(:param); $(:pertC) [$(:species)]")
@transform!(pert_kC_df, @byrow :condition = replace(:condition, "; 1x [$(pert_C.pattern)]"=>""))
@eachrow! pert_kC_df begin
	if (:pertk=="1x" && :pertC=="1x")
		:condition = "base model"
		:param = "base model"
		:species = "base model"
	end
end
@subset!(pert_kC_df, .!(:pertk.=="1x" .&& :pertC.=="10x"))
# @subset!(pert_kC_df, .!(:pertk.=="1x" .&& :pertC.=="100x"))
unique!(pert_kC_df)


## Plots
# -- Plot GAB1-SHP2:
kC_ord = unique(pert_kC_df.condition)[[1,3,5,2,4]]
fig_pert_Ck1 = @pipe pert_kC_df |> 
	# data(_) * aog.mapping(:r, :PG1S, color=:param, linestyle=:condition=>sorter(kC_ord)) * 
	data(_) * aog.mapping(:r, :PG1S, color=:condition=>sorter(kC_ord), linestyle=:condition=>sorter(kC_ord)) * 
	# data(_) * aog.mapping(:r, :PG1S, color=:condition, linestyle=:condition) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth, 
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="",
		limits=(0,R, 0,1)
		)
	)
fig_pert_Ck1 |> display
save("images/steady-state_PG1S_pert-$(pert_knames)_$(pert_Cname).pdf", fig_pert_Ck1, pt_per_unit=1)

# -- Plot pGAB1:
fig_pert_Ck2 = @pipe pert_kC_df |> 
	data(_) * aog.mapping(:r, :PG1, color=:condition=>sorter(kC_ord), linestyle=:condition=>sorter(kC_ord)) * 
	# data(_) * aog.mapping(:r, :PG1, color=:condition, linestyle=:condition) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth, 
		xlabel="r (μm)", ylabel="norm. pGAB1", title="",
		limits=(0,R,0,1)
		)
	)
fig_pert_Ck2 |> display
save("images/steady-state_pGAB1_pert-$(pert_knames)_$(pert_Cname).pdf", fig_pert_Ck2, pt_per_unit=1)










## =============== Run simulations when changing diffusivities and kinetic parameters ============== ##
# -- Kinetic parameter to change:
# pert_k = [r"kSi", r"kG1dp"]        # regex for rate constant of interest
pert_k = ["kS2r", "kG1dp"]        # regex for rate constant of interest
# pert_k = ["kS2f", "kG1p"]        # regex for rate constant of interest
pert_knames = join(pert_k,"-")
# pert_vec = [1., 100.]      # vector of multipliers for parameter perturbations
pert_vec = [1., 0.01]      # vector of multipliers for parameter perturbations

# -- Diffusivity to change:
pert_D = r"Dsfk"       # regex for initial protein concentration of interest
pert_Dind = match.(pert_D, Dnames) .!= nothing
pert_Dname = Dnames[pert_Dind][1]
# pert_vecD = [1., 10.]
pert_vecD = [1., 0.01]


pert_kD_df = DataFrame()
for j in eachindex(pert_vecD)
	Diffsⱼ = copy(Diffs)
    Diffsⱼ[pert_Dind] = Diffsⱼ[pert_Dind] * pert_vecD[j]
	for k in eachindex(pert_k)
		pert_kindₖ = occursin.(pert_k[k], knames)  # get which entry in name vector matches desired parameter name
		pert_knameₖ = knames[pert_kindₖ][1]
		Threads.@threads for i in eachindex(pert_vec)
			# Generate solution for perturbation i:
			kvalsᵢ = copy(kvals) 
			kvalsᵢ[pert_kindₖ] = kvalsᵢ[pert_kindₖ] * pert_vec[i]       # perturb kinetic rate constant
			@time ss, r_ss = pdesolver(Co, Diffsⱼ, kvalsᵢ; R=R, dr=dr, tf=tf, Nts=Nts) # Solve the model
			PG1S_ss = ss.PG1Stot[:,end]./maximum(ss.PG1Stot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
			PG1_ss = ss.PG1tot[:,end]./maximum(ss.PG1tot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
			dfᵢ = DataFrame(PG1S=PG1S_ss, PG1=PG1_ss, r=r_ss, 
				param=pert_knameₖ, D=pert_D.pattern,
				pertk=replace(string(pert_vec[i]), r".0$"=>"").*"x",
				pertD=replace(string(pert_vecD[j]), r".0$"=>"").*"x",
				)
			Threads.lock(lk) do
				append!(pert_kD_df, dfᵢ)
			end
		end
	end
end
@transform!(pert_kD_df, @byrow :condition = "$(:pertk) $(:param); $(:pertD) $(:D)")
# for i in pert_k @transform!(pert_kD_df, @byrow :condition = replace(:condition, "1x $(i);"=>"")) end
@transform!(pert_kD_df, @byrow :condition = replace(:condition, "; 1x $(pert_D.pattern)"=>""))
@eachrow! pert_kD_df begin
	if (:pertk=="1x" && :pertD=="1x")
		:condition = "base model"
		:param = "base model"
		:D = "base model"
	end
end
# @subset!(pert_kD_df, .!(:pertk.=="1x" .&& :pertD.!="1x"))
# @subset!(pert_kD_df, .!(:pertk.=="1x" .&& :pertD.=="10x"))
# @subset!(pert_kD_df, .!(:pertk.=="1x" .&& :pertC.=="100x"))
unique!(pert_kD_df)


## Plots
# -- Plot GAB1-SHP2:
# kD_ord = unique(pert_kD_df.condition)[[1,3,5,2,4]]
kD_ord = unique(pert_kD_df.condition)[[1,3,5,2,4,6,7]]
# kD_ord = unique(pert_kD_df.condition)[[1,3,2,4]]

fig_pert_Dk1 = @pipe pert_kD_df |> 
	# data(_) * aog.mapping(:r, :PG1S, color=:param, linestyle=:condition=>sorter(kD_ord)) * 
	data(_) * aog.mapping(:r, :PG1S, color=:condition=>sorter(kD_ord), linestyle=:condition=>sorter(kD_ord)) * 
	# data(_) * aog.mapping(:r, :PG1S, color=:condition, linestyle=:condition) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw*1.5, height=pert_plth*1.5, 
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title="",
		limits=(0,R, 0,1)
		)
	)
fig_pert_Dk1 |> display
save("images/steady-state_PG1S_pert-$(pert_knames)_$(pert_Dname).pdf", fig_pert_Dk1, pt_per_unit=1)

# -- Plot pGAB1:
fig_pert_Dk2 = @pipe pert_kD_df |> 
	data(_) * aog.mapping(:r, :PG1, color=:condition=>sorter(kD_ord), linestyle=:condition=>sorter(kD_ord)) * 
	# data(_) * aog.mapping(:r, :PG1, color=:condition, linestyle=:condition) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw*1.5, height=pert_plth*1.5, 
		xlabel="r (μm)", ylabel="norm. pGAB1", title="",
		limits=(0,R, 0,1)
		)
	)
fig_pert_Dk2 |> display
save("images/steady-state_pGAB1_pert-$(pert_knames)_$(pert_Dname).pdf", fig_pert_Dk2, pt_per_unit=1)





## =============== Steady-state calculations with just concentration perturbations =============== ##
#=
The following code is for obtaining model results at the final time tf when perturbing an initial
protein concentration by a factor of ten above and below its base value.
=#

# Initial species concentration to change:
# pert_C = r"SFK"       # regex for initial protein concentration of interest
pert_C = r"EGFR"       # regex for initial protein concentration of interest
# pert_C = r"SHP2"       # regex for initial protein concentration of interest
pert_Cind = match.(pert_C, Cnames) .!= nothing
pert_Cname = Cnames[pert_Cind][1]
# pert_vecC = [0.1, 1., 10.]      # vector of multipliers for parameter perturbations
# pert_vecC = [0.01, 1., 100.]      # vector of multipliers for parameter perturbations
# pert_vecC = [0.001, 1., 1000.]      # vector of multipliers for parameter perturbations
# pert_vecC = [0.01, 1.]      # vector of multipliers for parameter perturbations
# pert_vecC = [0.0001, 1.]      # vector of multipliers for parameter perturbations
pert_vecC = [0.001, 0.01, 0.1, 1.]      # vector of multipliers for parameter perturbations


pertC_df = DataFrame()
Threads.@threads for i in eachindex(pert_vecC)
	# Generate solution for perturbation i:
	Coᵢ = copy(Co)
	Coᵢ[pert_Cind] = Coᵢ[pert_Cind] * pert_vecC[i]       # perturb kinetic rate constant
	@time ss, r_ss = pdesolver(Coᵢ, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts) # Solve the model
	PG1S_ss = ss.PG1Stot[:,end]./maximum(ss.PG1Stot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	PG1_ss = ss.PG1tot[:,end]./maximum(ss.PG1tot[:,end])      # normalized distribution of pGAB1-SHP2 @ tf
	Threads.lock(lk) do
		dfᵢ = DataFrame(PG1S=PG1S_ss, PG1=PG1_ss, r=r_ss, param=pert_C.pattern, 
			pert = replace(string(pert_vecC[i]), r".0$"=>"") * "x [$(pert_C.pattern)]") 
		append!(pertC_df, dfᵢ)
	end
end

# -- Plot:
# pertC_ord = ["1x ","0.1x ","10x "] .* "[$(pert_C.pattern)]"
# pertC_ord = ["1x ","0.01x ","100x "] .* "[$(pert_C.pattern)]"
# pertC_ord = ["1x ","0.001x ","1000x "] .* "[$(pert_C.pattern)]"
# pertC_ord = ["1x ","$(pert_vecC[1])x ","$(Int(pert_vecC[3]))x "] .* "[$(pert_C.pattern)]"
# pertC_ord = ["1x ","$(pert_vecC[1])x "] .* "[$(pert_C.pattern)]"
pertC_ord = ["1x ","$(pert_vecC[3])x ","$(pert_vecC[2])x ","$(pert_vecC[1])x "] .* "[$(pert_C.pattern)]"
pertC_title = "[$(pert_C.pattern)] sensitivity"

# -- pGAB1-SHP2:
fig_pertC_pg1s = @pipe pertC_df |> 
    data(_) * mapping(:r, :PG1S, color=:pert=>sorter(pertC_ord), linestyle=:pert=>sorter(pertC_ord)) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth,
		xlabel="r (μm)", ylabel="norm. GAB1-SHP2", title=pertC_title,
		limits=(0,R,0,1)
		)
	)
fig_pertC_pg1s |> display
save("images/steady-state_PG1S_pert-$(pert_Cname).pdf", fig_pertC_pg1s, pt_per_unit=1)

# -- pGAB1:
fig_pertC_pg1 = @pipe pertC_df |> 
    data(_) * mapping(:r, :PG1, color=:pert=>sorter(pertC_ord), linestyle=:pert=>sorter(pertC_ord)) * 
		visual(Lines) |> 
	draw(_, axis=(width=pert_pltw, height=pert_plth,
		xlabel="r (μm)", ylabel="norm. pGAB1", title=pertC_title,
		limits=(0,R,0,1)
		)
	)
fig_pertC_pg1 |> display
save("images/steady-state_pGAB1_pert-$(pert_Cname).pdf", fig_pertC_pg1, pt_per_unit=1)







## =============== Model time scales =============== ##
τSi = ensemble[:,pnames.=="kSi"].^-1*60. |> vec
# τSa = (kSa*maximum(sol.E))^-1*60.
# τG1p = (kG1p*maximum(sol.aSFK))^-1*60.
τG1dp = ensemble[:,pnames.=="kG1dp"].^-1*60. |> vec
τEGFRp = kp^-1*60.
τEGFRdp = kdp^-1*60.
# τS2f = (kS2f*maximum(sol.PG1tot))^-1*60.
# τS2r = kS2r^-1*60.
τS2r = ensemble[:,pnames.=="kS2r"].^-1*60. |> vec
τDsfk  = R^2/(6*Diffs[1])*60.
τDg1s2  = R^2/(6*Diffs[6])*60.

GAB1dp_times = τDg1s2./τG1dp |> mean
SHP2r_times_quantiles = τG1dp ./ τS2r |> quantile
SHP2r_times = τG1dp ./ τS2r |> mean

println()
println("τSi = $(round.(median(τSi), sigdigits=3)) sec")
# println("τSa = $(round(τSa, sigdigits=3)) sec")
# println("τG1p = $(round(τG1p, sigdigits=3)) sec")
println("τG1dp = $(round.(median(τG1dp), sigdigits=3)) sec")
println("τEGFRp = $(round(τEGFRp, sigdigits=3)) sec")
println("τEGFRdp = $(round(τEGFRdp, sigdigits=3)) sec")
# println("τS2f = $(round(τS2f, sigdigits=3)) sec")
println("τS2r = $(round.(median(τS2r), sigdigits=3)) sec")
println("τDsfk = $(round(τDsfk, sigdigits=3)) sec")
println("τDg1s2 = $(round(τDg1s2, sigdigits=3)) sec")

println("GAB1 dp times: ", GAB1dp_times)
println("SHP2 unbinding times: ", SHP2r_times)
println("SHP2 unbinding times (quantiles): ", SHP2r_times_quantiles)

## =============== Model length scales =============== ##
# -- Calculate length scales using parameter ensembles:
ens_df = DataFrame(ensemble, pnames)
δ_SFK = sqrt.(ens_df.Dsfk./ens_df.kSi)
δ_dis = sqrt.(ens_df.Dg1s2./ens_df.kS2r)
δ_dep = sqrt.(ens_df.Dg1./ens_df.kG1dp)
δ_G1S2 = δ_dep .+ δ_dis .+ δ_SFK

## Plot length scales:
# -- Barplot:
fig_ls = @pipe DataFrame(δ_SFK=δ_SFK, δ_dis=δ_dis, δ_dep=δ_dep, δ_G1S2=δ_G1S2) |> 
	DataFrames.stack(_) |> 
	groupby(_, :variable) |> 
	@combine(_, :μ=median(:value), 
		:lci = quantile(:value, 0.5-0.341),	# calculating 68.2% credible interval (equivalent to ~1σ around median for a normal distribution)
		:uci = quantile(:value, 0.5+0.341)
		# :σ=std(:value), 
		# :sem=sem(:value)
		) |> 
	@transform(_, :lo=:μ .- :lci, :hi=:uci.-:μ) |> 
	# @subset(_, .!occursin.(r"G1S2", :variable)) |> 
	data(_) * mapping(:variable, :μ=>"Length (µm)") *
		(mapping(color=:variable)*visual(BarPlot) + mapping(:lo, :hi)*visual(Errorbars)) |> 
    draw(_, axis=(width=100, height=80, title="OOM estimates", 
        topspinevisible=false, rightspinevisible=false,
        xticklabelrotation=pi/6, limits=(nothing, (0,nothing)), #yticks=0:2:6,
        )
    )
fig_ls |> display
save("images/length scale comps/OOM_aSFK+G1S2_length_scales)barplot.png", fig_ls, px_per_unit=3)
save("images/length scale comps/OOM_aSFK+G1S2_length_scales)barplot.pdf", fig_ls, pt_per_unit=1)

# -- Boxplot:
fig_ls = @pipe DataFrame(δ_SFK=δ_SFK, δ_dis=δ_dis, δ_dep=δ_dep, δ_G1S2=δ_G1S2) |> 
	DataFrames.stack(_) |> 
	data(_) * mapping(:variable, :value=>"Length (µm)", color=:variable)*
		visual(BoxPlot, show_outliers=false) |> 
    draw(_, axis=(width=100, height=80, title="OOM estimates", 
        topspinevisible=false, rightspinevisible=false,
        xticklabelrotation=pi/6, limits=(nothing, (0,nothing)), #yticks=0:2:6,
        )
    )
fig_ls |> display
save("images/length scale comps/OOM_aSFK+G1S2_length_scales_boxplot.png", fig_ls, px_per_unit=3)
save("images/length scale comps/OOM_aSFK+G1S2_length_scales_boxplot.pdf", fig_ls, pt_per_unit=1)