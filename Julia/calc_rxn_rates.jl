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


function dxdt(x,t)
    dxdt = zeros(length(x))
    for i in eachindex(dxdt)
        if i == 1
            dxdt[i] = (x[2]-x[1]) / (t[2]-t[1])
        elseif i == length(dxdt)
            dxdt[i] = (x[end]-x[end-1]) / (t[end]-t[end-1])
        else
            dxdt[i] = (x[i+1]-x[i-1]) / (2.0*(t[2]-t[1]))
        end
    end
    return dxdt
end

## Include model solver function in current session:
include("basepdesolver.jl") # load finite-diff PDE solver
include("get_param_posteriors.jl")


## Define parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]



## Define solver and spatiotemporal grid constants:
dr = 0.25 # length of each spatial step
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
N = 5000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
retro_full_df = run_ensemble(pdesolver, ensemble, Co; dr=dr, tol=1e-2)


## ===== Calculate average rates of SFK activation and GAB1 phosphorylation ===== ##
v_df = DataFrame()
for i in axes(retro_full_df,1)
    rsolᵢ = retro_full_df.r[i] 
    tsolᵢ = retro_full_df.t_sol[i] 
    solᵢ = retro_full_df.sol[i]
    idxᵢ = retro_full_df.index[i]

    ksiᵢ = ensemble[idxᵢ, pnames.=="kSi"]
    ksaᵢ = ensemble[idxᵢ, pnames.=="kSa"]
    kg1dpᵢ = ensemble[idxᵢ, pnames.=="kG1dp"]
    kg1pᵢ = ensemble[idxᵢ, pnames.=="kG1p"]


    ## Calculate spatially and temporally averaged SFK quantities of interest:
    # -- Average rate of SFK inactivation:
    v_sfk_iᵢ = [NumericalIntegration.integrate(rsolᵢ, ksiᵢ .* solᵢ.aSFK[:,j] .* rsolᵢ.^2) for j in eachindex(tsolᵢ)] .* 3.0 ./ R^3  # spatially averaged
    v_sfk_i_aveᵢ = NumericalIntegration.integrate(tsolᵢ, v_sfk_iᵢ) ./ tf    # temporally averaged
    
    # -- Average rate of SFK activation:
    v_sfk_aᵢ = NumericalIntegration.integrate(tsolᵢ, ksaᵢ .* solᵢ.pE./(100.0./CoEGFR) .* solᵢ.iSFK[1,:]) ./ tf  # temporally averaged
    
    # -- Average net rate of SFK activation (derivative of aSFK w/r/t time):
    asfk_aveᵢ = [NumericalIntegration.integrate(rsolᵢ, solᵢ.aSFK[:,j] .* rsolᵢ.^2) for j in eachindex(tsolᵢ)] .* 3.0 ./ R^3
    v_sfk_netᵢ = NumericalIntegration.integrate(tsolᵢ, dxdt(asfk_aveᵢ, tsolᵢ)) ./ tf



    ## Calculate spatially and temporally averaged pGAB1 quantities of interest:
    # -- Spatially averaged total pGAB1 levels:
    G1_cytᵢ = solᵢ.GAB1 .+ solᵢ.G2G1
    PG1_cytᵢ = [NumericalIntegration.integrate(rsolᵢ, solᵢ.PG1tot[:,j] .* rsolᵢ.^2) for j in eachindex(tsolᵢ)] .* 3.0 ./ R^3
    PG1_totᵢ = PG1_cytᵢ .+ (solᵢ.EG2PG1 .+ solᵢ.EG2PG1S)./surfCF.*volCF
    
    # -- Average rate of pGAB1 dephosphorylation:
    v_pg1_dpᵢ = [NumericalIntegration.integrate(rsolᵢ, kg1dpᵢ .* solᵢ.PG1tot[:,j] .* rsolᵢ.^2) for j in eachindex(tsolᵢ)] .* 3.0 ./ R^3  # spatially averaged
    v_pg1_dp_aveᵢ = NumericalIntegration.integrate(tsolᵢ, v_pg1_dpᵢ) ./ tf    # temporally averaged
    
    # -- Average rate of GAB1 phosphorylation:
    v_g1_pᵢ = [NumericalIntegration.integrate(rsolᵢ, kg1pᵢ .* solᵢ.aSFK[:,j] .* G1_cytᵢ[:,j] .* rsolᵢ.^2) for j in eachindex(tsolᵢ)] .* 3.0 ./ R^3  # spatially averaged
    v_g1_p_aveᵢ = NumericalIntegration.integrate(tsolᵢ, v_g1_pᵢ) ./ tf    # temporally averaged
    
    # -- Average net rate of GAB1 phosphorylation (derivative of pg1 w/r/t time):
    v_pg1_netᵢ = NumericalIntegration.integrate(tsolᵢ, dxdt(PG1_totᵢ, tsolᵢ)) ./ tf  # temporally averaged net GAB1 phos. rate


    dfᵢ = DataFrame(v_sfka=v_sfk_aᵢ, v_sfki=v_sfk_i_aveᵢ, v_sfk_net=v_sfk_netᵢ,
        v_pg1=v_g1_p_aveᵢ, v_dpg1=v_pg1_dp_aveᵢ, v_pg1_net=v_pg1_netᵢ
        )

    append!(v_df, dfᵢ)
end

##
println()
println("QUANTILES FOR SFK QUANTITIES (0.025, 0.25, 0.5, 0.75, 0.975):")
println("Net SFK activation rate (µM/min):")
quantile(v_df.v_sfk_net) |> display
println()

println("Time-averaged SFK activation rate (µM/min):")
quantile((v_df.v_sfka .* (1e15/6.022e23*1e6))) |> display
println()

println("Time-averaged SFK inactivation rate (µM/min):")
quantile((v_df.v_sfki .* (1e15/6.022e23*1e6))) |> display
println()

println("1/Vsfk,a (1/(µM⋅min⁻¹)):")
quantile(1.0 ./ (v_df.v_sfka .* (1e15/6.022e23*1e6))) |> display
println()

println("1/Vsfk,i (1/(µM⋅min⁻¹)):")
quantile(1.0 ./ (v_df.v_sfki .* (1e15/6.022e23*1e6))) |> display
println()


println()
println("QUANTILES FOR GAB1 QUANTITIES (0.025, 0.25, 0.5, 0.75, 0.975):")
println("Net GAB1 phos. rate (µM/min):")
quantile(v_df.v_pg1_net) |> display
println()

println("Time-averaged GAB1 phos. rate (µM/min):")
quantile((v_df.v_pg1 .* (1e15/6.022e23*1e6))) |> display
println()

println("Time-averaged pGAB1 dephos. rate (µM/min):")
quantile((v_df.v_dpg1 .* (1e15/6.022e23*1e6))) |> display
println()

println("1/Vg1p (1/(µM⋅min⁻¹)):")
quantile(1.0 ./ (v_df.v_pg1 .* (1e15/6.022e23*1e6))) |> display
println()

println("1/Vg1dp (1/(µM⋅min⁻¹)):")
quantile(1.0 ./ (v_df.v_dpg1 .* (1e15/6.022e23*1e6))) |> display
println()