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
using CSV, DelimitedFiles, NumericalIntegration, Interpolations
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
include("pulsechase_solver.jl")
include("get_param_posteriors.jl")


## Load dynamic model data for comparison:
dynamic_fn = "pEGFR_pulsechase-res_dynamic.tsv"   # file path for ODE model results
pEGFR_dyn = readdlm(dynamic_fn) |> vec
t_dyn = collect(4.97:0.07:7)



## Model parameters and constants:
# -- Parameter names:
Dnames = ["Dsfk", "Dg2", "Dg2g1", "Dg2g1s2", "Dg1", "Dg1s2", "Ds2"]
knames = ["kS2f", "kS2r", "kG1f", "kG1r", "kG2f", "kG2r", "kG1p", "kG1dp",
    "kSa", "kSi", "kp", "kdp", "kEGFf", "kEGFr",  "EGF", "kdf", "kdr"]


# -- Solver and spatiotemporal grid constants:
dr = 0.1 # length of each spatial step
R = 10.0 #  Radius of cell
Nts = 120 #  Number of time points to store
t_prechase = 5.0    # EGF pre-chase time (min)
t_chase = 2.0   # gefitinib chase time (min)
tf = t_prechase + t_chase   # total simulation time
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



## Call PDE solver:
# @time sol, r_sol, t_sol = pulsechase_solver(Co, Diffs, kvals; R=R, dr=dr, t_prechase=t_prechase, tf=tf, Nts=Nts)     # solve model
# aSFK = sol.aSFK
# aSFK_chase = aSFK[:,t_sol.>=t_prechase] # aSFK during chase time
# PG1Stot = sol.PG1Stot
# PG1S_chase = PG1Stot[:,t_sol.>=t_prechase]  # total pGAB1-SHP2 during chase time
# pEGFR = sol.pE
# t_chase = t_sol[t_sol.>=t_prechase] .- t_prechase   # time with gefitinib (chase time)
# tdiff = t_chase[2]-t_chase[1]



## ===== Run ensemble calculations ===== ##
N = 2000		# number of parameter sets to use in ensemble
ensemble, D_ens, k_ens = generate_ensemble(chain, uvpars, mvpars, pnames, N=N)
res_df = run_ensemble_pc(pulsechase_solver, ensemble, Co; t_prechase=t_prechase, t_chase=t_chase, Nts=Nts)



## ===== Compute "summary" surfaces of interest ===== ##
r_med = res_df.r[1]	# spatial grid from ensemble calculations

## Extract surfaces for cytosolic GAB1-SHP2 and calculate median surface:
PG1S_stack = @pipe [(res_df.sol[i].PG1S .+ res_df.sol[i].G2PG1S) for i in axes(res_df,1)] |> 
    hcat(_...) |>   # horizontal concatenation
    reshape(_, (size(res_df.sol[1].PG1S)..., :)) # reshape to 3D matrix

PG1S_med0 = median(PG1S_stack, dims=3)[:,:,1]   # median GAB1-SHP2 surface
PG1S_lo0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5-0.341) for i in axes(PG1S_stack,1)] # lower credible interval @ t=tf
PG1S_up0 = [quantile(PG1S_stack[:,end,:][i,:], 0.5+0.341) for i in axes(PG1S_stack,1)] # upper credible interval @ t=tf

PG1S_lo_r = [quantile(PG1S_stack[end,:,:][i,:], 0.5-0.341) for i in axes(PG1S_stack,2)] # lower credible interval @ r=R
PG1S_up_r = [quantile(PG1S_stack[end,:,:][i,:], 0.5+0.341) for i in axes(PG1S_stack,2)] # upper credible interval @ r=R

# -- Interpolate values on dr=0.1 grid:
PG1S_med = zeros(length(rvec), Nts+1)
for i in axes(PG1S_med,2)
    interp_i = linear_interpolation(r_med, PG1S_med0[:,i])
    PG1S_med[:,i] = interp_i(rvec)
end

# -- Interpolate upper/lower CIs @ t=tf:
pg1s_inter_lo = linear_interpolation(r_med, PG1S_lo0)
pg1s_inter_up = linear_interpolation(r_med, PG1S_up0)

PG1S_lo = pg1s_inter_lo(0:0.1:R)  # interpolate lower CI
PG1S_up = pg1s_inter_up(0:0.1:R)  # interpolate upper CI




## =============== Plot pulse-chase results =============== ##
# ftsz = 8    # font size
# lw = 1      # line width
# xticks1 = [0.,1.0,2.0]
# xticks2 = [0.,R/2,R]

# ## Plot pEGFR timecourse:
# t_plot = [t_dyn .- t_dyn[1], t_chase]
# pEGFR_plot = [pEGFR_dyn./maximum(pEGFR_dyn), (pEGFR./maximum(pEGFR))[t_sol.>=t_prechase]]
# Plots.plot(t_plot, pEGFR_plot, label=["reaction-only model" "reaction-diffusion model"], ls=:auto, lw=lw, legend=:outerright,
#     tickdirection=:out, grid=false, xlims=(0,Inf), ylims=(0,Inf), xticks=xticks1, #yticks=yticks, 
#     tickfontsize=ftsz, labelfontsize=ftsz, legendfontsize=ftsz*0.75, titlefontsize=ftsz,
#     xlabel="min gefitinib", ylabel="norm. pEGFR",title="5 min EGF pre-chase",
#     size=(275,140)
#     ) |> display
# savefig("images/Pulse-chase_pEGFR.png")
# savefig("images/Pulse-chase_pEGFR.pdf")


# ## pGAB1-SHP2:
# # -- Snapshots:
# times = collect(0.0:0.5:2.0)
# t_plot2 = [t_chase[times[i]-tdiff .< t_chase .< times[i]+tdiff][1] for i in eachindex(times)]
# PG1S_plot = PG1S_chase[:, in(t_plot2).(t_chase)]; PG1S_plot = PG1S_plot./PG1S_plot[:,1]
# labels = ["0 min chase" "0.5 min chase" "1 min chase" "1.5 min chase" "2 min chase"]
# Plots.plot(r_sol, PG1S_plot, label=labels, ls=:auto, lw=lw, legend=:outerright,
#     tickdirection=:out, grid=false, xlims=(0,Inf), ylims=(0,Inf), xticks=xticks2, #yticks=yticks,
#     tickfontsize=ftsz, labelfontsize=ftsz, legendfontsize=ftsz*0.75, titlefontsize=ftsz,
#     xlabel="r (μm)", ylabel="norm. GAB1-SHP2",title="5 min EGF pre-chase",
#     size=(250,150)
#     ) |> display
# savefig("images/Pulse-chase_pGAB1-SHP2.png")
# savefig("images/Pulse-chase_pGAB1-SHP2.pdf")


# ## Full spatiotemporal dynamics:
# # -- Heatmap:
# hm_title = ""
# hm_title = "EGF-gefinitib pulse chase"
# pg1s_hm_df = DataFrame(t=repeat(t_sol, inner=size(PG1Stot,1)), r=repeat(r_sol, outer=length(t_sol)), z=reshape(PG1Stot,:))
# pg1s_hm_plt = data(pg1s_hm_df) *
#             mapping(:t=>"Time (min)", :r=>"r (μm)", :z=>"GAB1-SHP2\n(molec·μm⁻³)") * visual(Heatmap, colormap=:turbo)
# pg1s_hm_fig = draw(pg1s_hm_plt, axis=(title=hm_title, xticks=0:1:7,
#     # topspinevisible=true, topspinecolor=:gray75, rightspinevisible=true, rightspinecolor=:gray75,
#     width=150, height=150))
# pg1s_hm_fig |> display
# save("images/pulse-chase_pGAB1-SHP2_heatmap.png", pg1s_hm_fig, px_per_unit = 3); # save PNG
# save("images/pulse-chase_pGAB1-SHP2_heatmap.pdf", pg1s_hm_fig, pt_per_unit = 1); # save PDF




# ## Surface with default plot angle:
# ax2 = (type=Axis3, width=300, height=300, #xreversed=true, yreversed=true,
#     xgridvisible=true, ygridvisible=true, zgridvisible=true, 
#     xlabel="Time (min)", ylabel="r (μm)", zlabel="GAB1-SHP2\n(molec·μm⁻³)", title="EGF-gefinitib pulse chase"
#     )

# x2 = t_sol[1:δt:end]
# y2 = r_sol[1:δr:end]
# z2 = PG1Stot'[1:δt:end,1:δr:end]

# fig2, axis2, plt2 = mk.surface(x2, y2, z2, colormap=cmap, shading=false, axis=ax2) 
# mk.wireframe!(axis2, x2, y2, z2, color=:black, linewidth=1.0)
# # fig2, axis2, plt2 = mk.surface(t_sol[t0_ind], r_sol[1:δr:end], PG1Stot'[t0_ind,1:δr:end], colormap=cmap, shading=false, axis=ax2) 
# # mk.wireframe!(axis2, t_sol[t0_ind], r_sol[1:δr:end], PG1Stot'[t0_ind,1:δr:end], color=:black, linewidth=1.0)
# axis2.limits = (floor(x2[1]), ceil(x2[end]), 0, R, 0, 100)
# Colorbar(fig2[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
# fig2 |> display
# save("images/pulse-chase_pGAB1-SHP2_surf.png", fig2, px_per_unit = 3); # save PNG
# save("images/pulse-chase_pGAB1-SHP2_surf.pdf", fig2, pt_per_unit = 1); # save PDF


## Surface with rotated angle to see the back end of the pulse chase:
# -- Surface plot settings:
δt = 1     # number of time points to skip for plotting
δr = 5      # number of spatial steps to skip for plotting
cmap = :turbo
surfw = 200; surfh = 200
t0_ind = tvec .>= t_prechase   # indices of time points to plot
lw = 1
# ax3 = (type=Axis3, azimuth=-pi/4, width=300, height=300, #xreversed=true, yreversed=true,
ax3 = (type=Axis3, azimuth=-1.9pi/3, elevation=0.18pi, width=surfw, height=surfh, #xreversed=true, yreversed=true,
    xgridvisible=true, ygridvisible=true, zgridvisible=true, 
    xlabel="Gefitinib chase\ntime (min)", ylabel="r (μm)", zlabel="GAB1-SHP2\n(molec·μm⁻³)", title="EGF-gefinitib pulse chase"
    )   # define axis for figure

# -- Data:
x3 = tvec[t0_ind][1:δt:end] .- t_prechase
y3 = rvec[1:δr:end]
z3 = PG1S_med'[t0_ind,1:δr:end][1:δt:end,:]

# -- Plot:
fig3, axis3, plt3 = mk.lines(repeat([t_chase], length(rvec)), rvec, PG1S_lo,    # credible interval projections @ t=tf
    linewidth=lw, color=:red, linestyle=:dash, axis=ax3)
mk.lines!(axis3, repeat([t_chase], length(rvec)), rvec, PG1S_up, 
    linewidth=lw, color=:red, linestyle=:dash)
    
mk.lines!(axis3, x3, repeat([R], length(x3)), PG1S_lo_r[t0_ind],    # credible interval projections @ r=R
    linewidth=lw, color=:red, linestyle=:dash)
mk.lines!(axis3, x3, repeat([R], length(x3)), PG1S_up_r[t0_ind], 
    linewidth=lw, color=:red, linestyle=:dash)

mk.surface!(axis3, x3, y3, z3, colormap=cmap, shading=false)    # median surface
mk.wireframe!(axis3, x3, y3, z3, color=:black, linewidth=0.8)
axis3.limits = (0,t_chase, 0,R, 0,40) 
axis3.zlabel="GAB1-SHP2\n(molec·μm⁻³)"
Colorbar(fig3[1,2], limits=(0,100), colormap=cmap, spinewidth=lw, width=surfh/20, height=surfw/3, label="% max. concentration")
fig3 |> display

save("images/pulse-chase_pGAB1-SHP2_surf-rotated.png", fig3, px_per_unit = 3); # save PNG
save("images/pulse-chase_pGAB1-SHP2_surf-rotated.pdf", fig3, pt_per_unit = 1); # save PDF




