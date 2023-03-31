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
include("get_param_posteriors.jl")  # may error → rerun if so → loads model parameter values and distributions 


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

# Define initial concentrations in molecules/μm³ or molecules/μm² (for
# surface species):
numSFK = 166000.0   # Combined copy number of SFKs -- only Src, Lyn, Fyn, Yes, Hck, and Blk have reported copy numbers in HeLa cells from Kulak et al.
numG2 = 628000.0    # HeLa copy number of GRB2
numG1 = 1530.0  # HeLa copy number of GAB1
numS2 = 3.0e5   # HeLa copy number of SHP2 (PTPN11)
numEGFR = 93000.0   # HeLa copy number of EGFR

CoSFK = numSFK*volCF    # Total concentration of SFK (molec/μm³)
CoG2 = numG2*volCF  # Total concentration of GRB2 (molec/μm³)
CoG1 = numG1*volCF # Total concentration of GAB1 (molec/μm³)
CoS2 = numS2*volCF  # Total concentration of SHP2 (molec/μm³)
CoEGFR = numEGFR*surfCF     # Total concentration of EGFR (molec/μm²)

Co_hela = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations -- HeLa abundances
Co = [6e5*volCF*ones(4); 6e5*surfCF]    # initial (flat) concentrations -- "base" model
Cnames = ["CoSFK", "CoG2", "CoG1", "CoSHP2", "CoEGFR"]   # Names of initial concentration variables


## Call PDE solver: Get baseline/median model predictions
@time (sol₀, r_sol₀, t_sol₀) = pdesolver(Co, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts)     # solve model -- base model abundances
@time (sol, r_sol, t_sol) = pdesolver(Co_hela, Diffs, kvals; R=R, dr=dr, tf=tf, Nts=Nts)     # solve model -- HeLa abundances



## Run ensemble calculations:
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
retro_df_base = run_ensemble(pdesolver, ensemble, Co)   # base model
retro_df_hela = run_ensemble(pdesolver, ensemble, Co_hela)   # HeLa model



## ===== Compute "summary" surfaces of interest ===== ##
r_med = retro_df_hela.r[1]	# spatial grid from ensemble calculations

# -- Extract surfaces for active SFKs and calculate "median" surface:
aSFK_stack = @pipe [retro_df_hela.sol[i].aSFK for i in axes(retro_df_hela,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_df_hela.sol[1].aSFK)..., :)) # reshape to 3D matrix

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
PG1S_stack = @pipe [retro_df_hela.sol[i].PG1Stot for i in axes(retro_df_hela,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_df_hela.sol[1].PG1Stot)..., :)) # reshape to 3D matrix

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


# -- Calculate average numbers of GAB1-SHP2 molecules @ t = 5 min:
g1s2_df = DataFrame()
lk = ReentrantLock()

Threads.@threads for j in axes(retro_df_hela,1)
	solⱼ = retro_df_hela.sol[j]
	r_solⱼ = retro_df_hela.r[j]
	PG1S_cytⱼ = solⱼ.PG1S .+ solⱼ.G2PG1S  # cytoplasmic GAB1-SHP2
    PG1S_cyt_aveⱼ = NumericalIntegration.integrate(r_solⱼ, PG1S_cytⱼ[:,end].*r_solⱼ.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
    PG1S_memⱼ = solⱼ.EG2PG1S[end] .* volCF/surfCF     # membrane GAB1-SHP2, converted to volume units
    # @show PG1S_memⱼ
    PG1Stotⱼ = PG1S_cyt_aveⱼ .+ PG1S_memⱼ     # total GAB1-SHP2
    num_PG1Sⱼ = PG1Stotⱼ[1]./volCF

	pct_dfⱼ = DataFrame(num_PG1S=num_PG1Sⱼ, type="model")
	Threads.lock(lk) do
		append!(g1s2_df, pct_dfⱼ)
	end
end
@combine(g1s2_df, :mean=mean(:num_PG1S), :std=std(:num_PG1S), :quantiles=quantile(:num_PG1S)) |> display



## Extract surfaces for cytosolic phospho-GAB1 and calculate median and upper/lower CI surfaces:
PG1_stack = @pipe [(retro_df_hela.sol[i].PG1 .+ retro_df_hela.sol[i].G2PG1) for i in axes(retro_df_hela,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_df_hela.sol[1].PG1)..., :)) |>  # reshape to 3D matrix
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
    xlabel="Time (min)", ylabel="r (μm)", title="HeLa protein abundances\n5 min EGF")
# update_theme!(fontsize=12, font="Arial")
update_theme!(fontsize=8, font="Arial")



## ===== Active SFKs ===== ##
fig1, axis1, plt1 = mk.lines(repeat([tf], length(tvec)), r_plt, aSFK_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis1, repeat([tf], length(tvec)), r_plt, aSFK_up, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis1, tvec[1:δt:end], r_plt[1:δr:end], aSFK_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis1.limits=(0,tf, 0,R, 0,nothing)
# axis1.xticks=[0,5]; #axis1.zticks=0:2e5:6e5
# axis1.zlabel="aSFK (cell⁻¹)"
axis1.zlabel="aSFK\n(molec·μm⁻³)"
Colorbar(fig1[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig1 |> display
save("images/aSFK_median-surf_N=$(N)-spls_HeLa.png", fig1, px_per_unit=3)
save("images/aSFK_median-surf_N=$(N)-spls_HeLa.pdf", fig1, pt_per_unit=1)




## ===== Total pGAB1 ===== ##
fig2, axis2, plt2 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis2, repeat([tf], length(tvec)), r_plt, PG1_up, 
    linewidth=lw, color=:red, linestyle=:dash)
mk.surface!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], colormap=cmap, interpolate=false, shading=false) 
mk.wireframe!(axis2, tvec[1:δt:end], r_plt[1:δr:end], PG1_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis2.limits=(0,tf, 0,R, 0,nothing)
# axis2.xticks=[0,5]; #axis2.zticks=0:2e5:6e5
# axis2.zlabel="pGAB1 (cell⁻¹)"
axis2.zlabel="pGAB1\n(molec·μm⁻³)"
Colorbar(fig2[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig2 |> display
save("images/pGAB1_median-surf_N=$(N)-spls_HeLa.png", fig2, px_per_unit=3)
save("images/pGAB1_median-surf_N=$(N)-spls_HeLa.pdf", fig2, pt_per_unit=1)




## ===== Total pGAB1-SHP2 ===== ##
fig3, axis3, plt3 = mk.lines(repeat([tf], length(tvec)), r_plt, PG1S_lo,
    linewidth=lw, color=:red, linestyle=:dash, axis=ax_surfs)
mk.lines!(axis3, repeat([tf], length(tvec)), r_plt, PG1S_up, 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], colormap=cmap, shading=false) 
mk.wireframe!(axis3, tvec[1:δt:end], r_plt[1:δr:end], PG1S_med'[1:δt:end,1:δr:end], color=:black, linewidth=lw)
axis3.limits = (0,tf, 0,R, 0,nothing) 
# axis3.xticks=[0,5]; #axis3.zticks=0:1.5e5:4.5e5
# axis3.zlabel="pGAB1-SHP2 (cell⁻¹)"
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls_HeLa.png", fig3, px_per_unit=3)
save("images/pGAB1-SHP2_median-surf_N=$(N)-spls_HeLa.pdf", fig3, pt_per_unit=1)






## =============== Calculate and plot length scale metrics of interest =============== ##
# -- Base model pGAB1-SHP2 ensemble predictions:
base_stack = @pipe [retro_df_base.sol[i].PG1Stot for i in axes(retro_df_base,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(retro_df_base.sol[1].PG1S)..., :)) # reshape to 3D matrix

# -- Center:surface ratio of GAB1-SHP2 @ tf:
csr_hela = DataFrame(value=PG1S_stack[1,end,:]./PG1S_stack[end,end,:], model="HeLa")
csr_base = DataFrame(value=base_stack[1,end,:]./base_stack[end,end,:], model="Base")


## Compute Bayes factor for difference between distributions:
# -- Load RCall and R package for calculating Bayes factor:
using RCall
@rlibrary ggstatsplot
@rlibrary BayesFactor	# for getting Bayes factors as a variable
@rlibrary easystats		# for interpreting Bayes factors
x = [csr_hela.model; csr_base.model]
y = [csr_hela.value; csr_base.value]

# -- Calculate Bayes factor for difference between distributions:
R"
ggstatsplot::ggbetweenstats(data.frame(x=$x, y=$y), x, y, 
	paired=TRUE,
	)
"

R"""
BFtest <- BayesFactor::ttestBF($(csr_hela.value), $(csr_base.value))	# compute Bayes factor
BF <- BayesFactor::extractBF(BFtest)	# extract Bayes factor
print(effectsize::interpret_bf(BF$bf))	# assess evidence for alternative hypothesis (diff. b/w means > 0)
"""

BF_res = @rget(BFtest); print(BF_res)	# get and print Bayes factor results
println("ln(BF) = $(log.(@rget(BF).bf)[1])")	# ln(BF)



## Plot comparisons:
# -- Barplot:
fig_csr_comp = @pipe vcat(csr_hela, csr_base) |> 
    groupby(_, :model) |> 
    @combine(_, :value=median(:value), :lci=quantile(:value, 0.5-0.341), :hci=quantile(:value, 0.5+0.341)) |> 
    # @combine(_, :value=median(:value), :lci=quantile(:value, 0.055), :hci=quantile(:value, 1-0.055)) |> 
    @transform(_, :lo=:value.-:lci, :hi=:hci.-:value) |> 
	data(_) * 
		mapping(:model=>"Protein abundances", :value=>"Cg1s2(0)/Cg1s2(R)") * 
			(mapping(color=:model)*visual(BarPlot) + mapping(:lo, :hi)*visual(Errorbars)) |> 
    draw(_, axis=(width=70, height=80, limits=(nothing,(0.,1.)), title="+EGF, 5 min"))
fig_csr_comp |> display
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_HeLa.png", fig_csr_comp, px_per_unit=3)
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_HeLa.pdf", fig_csr_comp, pt_per_unit=1)


# -- Rainclouds plot:
fig_csr_comp2 = @pipe vcat(csr_hela, csr_base) |> 
    data(_) * 
        mapping(:model, :value=>"Cg1s2(0)/Cg1s2(R)", color=:model) * 
            visual(RainClouds, gap=0, violin_limits=(0,1)) |> 
    draw(_, axis=(;height=120, width=100, limits=(nothing, (0,1)), title="+EGF, 5 min")) 
fig_csr_comp2 |> display
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_rainclouds_HeLa.png", fig_csr_comp2, px_per_unit=3)
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_rainclouds_HeLa.pdf", fig_csr_comp2, pt_per_unit=1)


# -- Boxplot:
fig_csr_comp3 = @pipe vcat(csr_hela, csr_base) |> 
    data(_) * 
        mapping(:model, :value=>"Cg1s2(0)/Cg1s2(R)", color=:model) * 
            visual(BoxPlot, show_outliers=false) |> 
    draw(_, axis=(;height=120, width=100, limits=(nothing, (0,1)))) 
fig_csr_comp3 |> display
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_boxplot_HeLa.png", fig_csr_comp3, px_per_unit=3)
save("images/length scale comps/GAB1-SHP2_center-to-surface_ratio_comps_boxplot_HeLa.pdf", fig_csr_comp3, pt_per_unit=1)