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
include("get_param_posteriors.jl")  # if errors, try re-running


## Define solver and spatiotemporal grid constants:
dr = 0.01   # desired spatial step size for interpolated solution grid
dr_ens = 0.4   # length of spatial step for ensemble simulations
R = 100.0 #  Radius of cell
Nts = 2 #  Number of time points to store
tf = 5.0 # final integration time
tvec = collect(0:tf/Nts:tf)
rvec = collect(0:dr:R)

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


## ===== Generate parameter ensemble ===== ##
# -- Parameters to perturb for length scale calculations:
p_change = ["Dsfk", "Dg1", "Dg1s2", "kSi", "kG1dp", "kS2r"]
p_mult = [0.1, 1.0, 10.0]

# -- Generate parameter ensemble:
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)


## Run ensemble simulations:
res_df = DataFrame()    # DataFrame for results
lk = ReentrantLock()
for param in p_change
	for mult in p_mult
		ensⱼ = copy(ensemble)
		ensⱼ[:,pnames.==param] *= mult	# modify parameter(s) of interest
        ens_dfⱼ = DataFrame(ensⱼ, pnames)    # DataFrame version, for OOM estimates
		retro_full_df = run_ensemble(pdesolver, ensⱼ, Co; R=R, dr=dr_ens, Nts=Nts)	# ensemble runs

		# -- Calculate fraction of EGFR-bound GAB1-SHP2 complexes and complex center:surface ratio:
		Threads.@threads for j in axes(retro_full_df,1)
			solⱼ = retro_full_df.sol[j]
			r_solⱼ = retro_full_df.r[j]
			aSFKⱼ = solⱼ.aSFK[:,end]  # cytoplasmic GAB1-SHP2
			PG1S_cytⱼ = (solⱼ.PG1S .+ solⱼ.G2PG1S)[:,end]  # cytoplasmic GAB1-SHP2

            # -- Interpolate on dr=0.1 grid:
            asfk_interpⱼ = linear_interpolation(r_solⱼ, aSFKⱼ)
            pg1s_interpⱼ = linear_interpolation(r_solⱼ, PG1S_cytⱼ)

            aSFK = asfk_interpⱼ(rvec)
            PG1Stot = pg1s_interpⱼ(rvec)

			# PG1S_memⱼ = solⱼ.EG2PG1S[end] .* volCF/surfCF     # membrane GAB1-SHP2, converted to volume units
			# ratioⱼ = PG1S_cytⱼ[1,end]/PG1S_cytⱼ[end,end]	# center:surface ratio

            # -- Length scale calculations:
            r12_sfk = R - (rvec[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
            r110_sfk = R - (rvec[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
            r12_pg1s = R - (rvec[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
            r110_pg1s = R - (rvec[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

            # -- Order-of-magnitude estimates:
            δ_SFK = sqrt(ens_dfⱼ.Dsfk[j]/ens_dfⱼ.kSi[j])
            δ_dis = sqrt(ens_dfⱼ.Dg1s2[j]/ens_dfⱼ.kS2r[j])
            δ_dep = sqrt(ens_dfⱼ.Dg1[j]/ens_dfⱼ.kG1dp[j])
            δ_G1S2 = δ_dep + δ_dis + δ_SFK
			
			if isnan(r110_pg1s)	# skip if NaN
			else
                # -- SFK results:
                sfk_dfⱼ = DataFrame(id=j, param=param, mult=mult,
                    r12=r12_sfk,
                    r110=r110_sfk,
                    δ=δ_SFK,
                    species="aSFK"
                    )
                # -- GAB1-SHP2 results:
				pg1s_dfⱼ = DataFrame(id=j, param=param, mult=mult,
                    r12=r12_pg1s,
                    r110=r110_pg1s,
                    # δ_SFK=δ_SFK,
                    # δ_dis=δ_dis,
                    # δ_dep=δ_dep,
                    # δ_G1S2=δ_G1S2,
                    δ=δ_G1S2,
                    species="GAB1-SHP2"
                    )
                # -- Combine results:
                res_dfⱼ = vcat(sfk_dfⱼ, pg1s_dfⱼ)
				Threads.lock(lk) do
					append!(res_df, res_dfⱼ)
				end
			end
		end
	end
end
@transform!(res_df, :mult=string.(:mult).*"x")
@transform!(res_df, :mult=replace.(:mult, ".0x"=>"x"))




## =============== Plot length scale estimates =============== ##
xord = ["0.1x","1x","10x"]  # order of perturbations for plotting



## Comparisons of model and OM estimates:
# -- aSFK r1/2 comparisons:
pltw = 90 #60
plth = 70
spec_ord = ["aSFK","GAB1-SHP2"]

fig_bar = @pipe res_df |>
    DataFrames.stack(_, Not([:id, :param, :mult, :species])) |> 
    groupby(_, [:param, :mult, :species, :variable]) |> 
    @combine(_, :med=mean(:value), 
        :lci=quantile(:value, 0.5-0.341), 
        :uci=quantile(:value, 0.5+0.341)
        ) |>
    @transform(_, :lo=:med-:lci, :hi=:uci-:med) |> 
    @subset(_, .!occursin.("r110",:variable)) |> 
    # @subset(_, :species.=="aSFK", .!occursin.("r110",:variable)) |> 
    data(_) *
        mapping(:mult=>sorter(xord), :med, 
            dodge=:variable, col=:param, row=:species=>sorter(spec_ord)) * 
            (mapping(color=:variable)*visual(BarPlot) + 
                mapping(:lo, :hi)*visual(Errorbars)) |> 
    draw(_, 
        axis=(xlabel="Parameter value", ylabel="Distance from\ncell membrane (μm)", 
            width=pltw, height=plth, limits=(nothing,nothing,0,nothing)
        ),
        facet=(;linkxaxes=:none,linkyaxes=:none)
    )
fig_bar |> display
save("images/length scale comps/OOM-model_aSFK+G1S2_length-scale-comps_R=$(R)_barplot.png", fig_bar, px_per_unit=3)
save("images/length scale comps/OOM-model_aSFK+G1S2_length-scale-comps_R=$(R)_barplot.pdf", fig_bar, pt_per_unit=1)


fig_box = @pipe res_df |>
    DataFrames.stack(_, Not([:id, :param, :mult, :species])) |> 
    @subset(_, .!occursin.("r110",:variable)) |> 
    # @subset(_, :species.=="aSFK", .!occursin.("r110",:variable)) |> 
    data(_) *
        mapping(:mult=>sorter(xord), :value, color=:variable, 
            dodge=:variable, col=:param, row=:species=>sorter(spec_ord)) * 
            visual(BoxPlot, show_outliers=false) |> 
    draw(_, 
        axis=(xlabel="Parameter value", ylabel="Distance from\ncell membrane (μm)", 
            width=pltw, height=plth, limits=(nothing,nothing,0,nothing)
        ),
        facet=(;linkxaxes=:none,linkyaxes=:none),
        # palettes=(;color=aog.wongcolors()[3:4])
    )
fig_box |> display
save("images/length scale comps/OOM-model_aSFK+G1S2_length-scale-comps_R=$(R)_boxplot.png", fig_box, px_per_unit=3)
save("images/length scale comps/OOM-model_aSFK+G1S2_length-scale-comps_R=$(R)_boxplot.pdf", fig_box, pt_per_unit=1)



##

ens_df = DataFrame(ensemble, pnames)
# ens_df.kS2r *= 0.01
# ens_df.Dg1s2 *= 100
δ_SFK = sqrt.(ens_df.Dsfk./ens_df.kSi)
δ_dis = sqrt.(ens_df.Dg1s2./ens_df.kS2r)
δ_dep = sqrt.(ens_df.Dg1./ens_df.kG1dp)
δ_G1S2 = δ_dep .+ δ_dis .+ δ_SFK
