#=
This script contains the code needed to estimate protein diffusivities based on data from Pepperkok et al. (DOI: 10.1083/jcb.111.6.3003)
=#
## =============== Load packages =============== ##
using Dierckx   # interpolation library
using Measurements  # for error propagation
using DataFrames, StatsBase
using CairoMakie, AlgebraOfGraphics


## =============== Pepperkok et al. data =============== ##
diff_cf = 1.0/(100.)^2. * (1e6)^2. * 60. # Diffusivity conversion factor from cm²/s --> μm²/min

# Rh-tubulin: mean of tubulin diffusivity values from Pepperkok et al. (DOI: 10.1083/jcb.111.6.3003) out to three decimal places
D_rhtub = mean(1e-8.*[1.61±0.10, 1.34±0.12])*diff_cf    # μm²/min
M_rhtub = 50000.     # molecular weight of tubulin, in Da


# Data from Erickson, 2009:
MW_std = [14044., 25665., 42910., 69322., 157368., 239656., 489324., 606444.] #, 387344]; # Molecular weights, in Da
Rs = [1.64, 2.09, 3.05, 3.55, 4.81, 5.20, 6.10, 8.50] #, 10.7]; # Stokes radii, in nm
Rs_uncertain = Rs .± 0.1.*Rs


## =============== Spline interpolation: Protein mass => Stokes radius =============== ##
#= 
Since cubic Hermite splines are not readily available from the major interpolation packages in Julia (the
one package implementing them, CubicHermiteSpline.jl, requires the gradient as an input, and Interpolations.jl
has not yet fully implemented them), we will instead use a QUADRATIC (2nd-order) spline for the time being since
the extrapolation beyond the endpoints is more suitable with a quadratic spline.

If a cubic Hermite spline is truly needed, then it may be possible to use something like finite differences or ForwardDiff
to construct a gradient function that can be used by CubicHermiteSpline.jl until a suitable implementation of Hermite splines with
just individual data points is implemented somewhere... Or just use a function from MATLAB, Python, or some other language (or make one yourself).
=#
# Construct spline w/ standards from Pepperkok et al.:
spline_order = 1    # spline order => anything from 1 to 5
itp = Spline1D(MW_std, Rs; k=spline_order, bc="extrapolate", s=0.0)    # construct spline
M_itp = LinRange(minimum(MW_std)*0.1, maximum(MW_std)*1.1, 1000)     # range over which to denerate spline points
Rs_itp = itp(M_itp)
Rs_tub = itp(M_rhtub)

# Calculate Stokes radii for proteins of interest:
# prot_names = ["SFK","GRB2","GAB1","SHP2","GRB2-GAB1","GAB1-SHP2","GRB2-GAB1-SHP2"] # proteins of interest
prot_names = ["Dsfk", "Dg2", "Dg1", "Ds2", "Dg2g1", "Dg1s2", "Dg2g1s2"]
Mi = [59835., 25206., 115000., 68436.]    # Molecular weight estimates of SFKs, GRB2, GAB1, and SHP2
MW = [Mi..., Mi[2]+Mi[3], sum(Mi[3:4]), sum(Mi[2:end])]     # Molecular weights of all cytosolic model species]   
Rs_est = itp(MW) # estimated Stokes radii from interpolation


## Plot spline w/ standards and proteins of interest:
# rs_plot = lines(M_itp, Rs_itp, axis=(type=Axis, xlabel="Molecular weight (Da)", ylabel="Stokes radius (nm)", title="Protein MW-Stokes radius interpolations"))
# scatter!(MW_std, Rs, label="Standards", ms=6)
# scatter!([M_rhtub], [Rs_tub], label="Rh-tubulin", ms=6, color=:red)
# for i in eachindex(prot_names)
#     scatter!([MW[i]], [Rs_est[i]], label=prot_names[i], ms=6, color=:orange)
# end
# display(rs_plot)



## =============== Estimate protein diffusivities =============== ##
#=
We now estimate protein diffusivities based on the diffusivity and Stokes radius of 
Rh-tubulin, including incorporating experimental uncertainty in the diffusivity of 
Rh-tubulin to obtain uncertainty estimates of the protein diffusivities of interest.
=#
# Protein standards and interpolated points:
Rs_rat_std = Rs_tub./Rs
D_std = D_rhtub*Rs_rat_std 

Rs_rat_itp = Rs_tub./Rs_itp
D_itp = D_rhtub*Rs_rat_itp

# Proteins of interest:
Rs_rat = Rs_tub./Rs_est     # Ratios of Rs for tubulin to Stokes radii for proteins of interest
D_est = D_rhtub.*Rs_rat     # Diffusivities of proteins of interest based on diffusivity of Rh-tubulin


## Plot diffusivity estimates:
# diff_plot = lines(M_itp, [i.val for i in D_itp], axis=(;xlabel="Molecular weight (Da)", 
#     ylabel="Diffusivity (μm²/min)", title="Protein diffusivity estimates"))
# scatter!(MW_std, [i.val for i in D_std], label="Standards", ms=6, shape=:auto)
# scatter!([M_rhtub], [D_rhtub.val], label="Rh-tubulin", ms=6, shape=:auto)
# for i in eachindex(prot_names)
#     scatter!([MW[i]], [D_est[i].val], label=prot_names[i], ms=6, shape=:auto)
# end
# display(diff_plot)

# Collect and display estimated diffusivities:
D_df = DataFrame(species=prot_names, D=round.(D_est))
# print(D_df)