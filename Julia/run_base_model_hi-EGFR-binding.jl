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
include("get_param_posteriors.jl")	# may error -- re-run if so



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



## Run ensemble calculations:
# -- Generate ensembles:
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)

# -- Parameter(s) we want to change:
# p_change = "kG2f"
# p_change = "kG1f"
# p_change = "kS2f"
# p_change = ["kG2f","kG1f","kdf","kp"]
p_change = ["kG2f"]
# p_mult = 10.0
# p_mult = 0.01
p_mult = 10.0.^(1:4)




## Ensemble simulations:
frac_df = DataFrame()
lk = ReentrantLock()
for param in p_change
	for mult in p_mult
		ensⱼ = copy(ensemble)
		ensⱼ[:,pnames.==param] *= mult	# modify parameter(s) of interest
		retro_full_df = run_ensemble(pdesolver, ensⱼ, Co)	# ensemble runs

		# -- Calculate fraction of EGFR-bound GAB1-SHP2 complexes and complex center:surface ratio:
		Threads.@threads for j in axes(retro_full_df,1)
			solⱼ = retro_full_df.sol[j]
			r_solⱼ = retro_full_df.r[j]
			PG1S_cytⱼ = solⱼ.PG1S .+ solⱼ.G2PG1S  # cytoplasmic GAB1-SHP2
			PG1S_cyt_aveⱼ = NumericalIntegration.integrate(r_solⱼ, PG1S_cytⱼ[:,end].*r_solⱼ.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
			PG1S_memⱼ = solⱼ.EG2PG1S[end] .* volCF/surfCF     # membrane GAB1-SHP2, converted to volume units
			# @show PG1S_memⱼ
			PG1Stotⱼ = PG1S_cyt_aveⱼ + PG1S_memⱼ     # total GAB1-SHP2
			frac_PG1Sⱼ = PG1S_memⱼ./PG1Stotⱼ	# fraction bound to EGFR
			ratioⱼ = PG1S_cytⱼ[1,end]/PG1S_cytⱼ[end,end]	# center:surface ratio
			
			if any(frac_PG1Sⱼ .< 0)	# skip if negatives
			elseif any(frac_PG1Sⱼ .> 1)	# skip if negatives
			elseif any(ratioⱼ .< 0)	# skip if negatives
			elseif any(ratioⱼ .> 1)	# skip if negatives
			elseif any(isnan.(frac_PG1Sⱼ))	# skip if NaNs
			else
				frac_dfⱼ = DataFrame(memb_frac=frac_PG1Sⱼ, csr=ratioⱼ, param=param, mult=mult)
				Threads.lock(lk) do
					append!(frac_df, frac_dfⱼ)
				end
			end
		end
	end
end
@transform!(frac_df, :mult = string.(Int.(:mult)).*"x") #.*"x ".*:param)


## ===== Plots of EGFR-bound SHP2 fraction versus GAB1-SHP2 center:surface ratio ===== ##
pltw = 80	# plot width
plth = 80	# plot height
ms = 5		# marker size
α = 0.25	# marker alpha/transparency

fig_bound = @pipe frac_df |> 
	# @transform(_, :mult = replace.(:mult, ".0x"=>"x")) |> 
	data(_) *
		mapping(:memb_frac=>"fraction EGFR-bound\nGAB1-SHP2", :csr=>"[G1S2](0)/[G1S2](R)", 
			color=:mult, col=:mult=>sorter(unique(frac_df.mult)), row=:param) * 
			# visual(Scatter, markersize=ms) |> 
			(visual(Scatter, markersize=ms) + linear()) |> 
	draw(_, axis=(width=pltw, height=plth, limits=(0,nothing,0,1.05),
			),
		facet=(;linkyaxes=:none, linkxaxes=:none)
		)
fig_bound |> display
save("images/changing-params_memb-G1S2_vs_cent-surf-ratio_t=$(Int(tf))min.png", fig_bound, px_per_unit=3)
save("images/changing-params_memb-G1S2_vs_cent-surf-ratio_t=$(Int(tf))min.pdf", fig_bound, pt_per_unit=1)

