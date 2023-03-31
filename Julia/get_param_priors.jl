using CSV, DataFrames, DataFramesMeta, Pipe, StatsBase
using Measurements  # for automatic propagation of uncertainties using the `Measurement` type
using Random

# -- Calculate protein diffusivities and estimated uncertainties
include("diffusivity_calculations.jl")  

# -- Functions for generating parameters of parameter prior distributions
include("param_distribution_funcs.jl")


## =============== Generate parameters for prior distributions =============== ##
# -- EGF concentration:
@everywhere EGF = 1.67e-3  # EGF concentration, μM -- equivalent to 10 ng/mL


#= See https://doi.org/10.1038/s41596-018-0056-z for a step-by-step protocol (implemented below) 
    for generating parameter prior distributions based off experimental/literature information. =#
expts_df = DataFrame()  # initialize DataFrame for holding exptl/prior measurements

# -- Exptl/prior information for EGFR-GRB2 binding:
kG2f_exptl = 16.0 *1e15*1e6/6.022e23*60.
KdG2_vals = [100.0] ./1e15/1e9*6.022e23
kG2f_0 = DataFrame(name="kG2f", vals=[[kG2f_exptl]], errs=[[1.1]], wts=[[12.]], err_types=[[1]])
kG2r_0 = DataFrame(name="kG2r", vals=[[480.]], errs=[[1.1]], wts=[[12.]], err_types=[[1]])
KdG2_0 = DataFrame(name="KdG2", vals=[KdG2_vals], errs=[[3.]], wts=[[12.]], err_types=[[1]])
append!(expts_df, kG2f_0)
append!(expts_df, kG2r_0)
append!(expts_df, KdG2_0)



## Exptl/prior information for SHP2-pGAB1: Using same as for GRB2 (simlar process → SH2 domain binding to pY)
kS2f_0 = DataFrame(name="kS2f", vals=[[kG2f_exptl]], errs=[[1.1]], wts=[[10.]], err_types=[[1]])
kS2r_0 = DataFrame(name="kS2r", vals=[[480.]], errs=[[1.1]], wts=[[10.]], err_types=[[1]])
KdS2_0 = DataFrame(name="KdS2", vals=[KdG2_vals], errs=[[3.0]], wts=[[10.]], err_types=[[1]])
append!(expts_df, kS2f_0)
append!(expts_df, kS2r_0)
append!(expts_df, KdS2_0)



## Exptl/prior information for GAB1-GRB2 binding: 
#= Data primarily from GRB2 and Src measurements in https://doi.org/10.1074/jbc.M501745200 → GRB2 and c-Src SH3 domains
    binding to proline-rich domains (PRDs) from dynamin 1/2 =#
# Forward binding rate:
kG1f_vals = [2.3e3, 6.4e4, 9.5e4, 1.1e3, 7.8e3, 1.5e4, 1.3e3, 2.4e4, 0.9e3] ./6.022e23 .* 1e15 .* 60.
kG1f_errors = [0.1e3, 0.1e4, 0.1e4, 7.0e3, 0.1e3, 0.2e4, 0.2e4, 0.3e4, 0.1e3]./6.022e23 .* 1e15 .* 60. .*sqrt(30.0)
kG1f_wts = [12., 12., 12., 12., 10., 10., 10., 10., 10.]

kG1f_err_types = zeros(length(kG1f_errors))
# Reverse binding rate:
kG1r_vals = [3.9e-2, 1.9e-3, 2.2e-3, 3.0e-3, 9.9e-4,  2.2e-3, 1.6e-3, 3.2e-3, 1.6e-3].*60.
kG1r_errors = [0.2e-2, 0.2e-3, 0.1e-3, 0.1e-3, 0.2e-4, 0.3e-3, 0.3e-3, 0.3e-3, 0.04e-3].*60. .*sqrt(30.0)

# Estimate KD values using forward/reverse rates (one for GRB2, one for Src from these data):
KdG1_vals = [mean(kG1r_vals[1:4])./mean(kG1f_vals[1:4]), mean(kG1r_vals[5:end])./mean(kG1f_vals[5:end])]
KdG1_errors = [mean((kG1r_vals[1:4] .± kG1r_errors[1:4])./mean(kG1f_vals[1:4] .± kG1f_errors[1:4])).err,
    mean((kG1r_vals[5:end] .± kG1r_errors[5:end])./mean(kG1f_vals[5:end] .± kG1f_errors[5:end])).err
    ]

# DataFrames that will be used as inputs to function for generating prior distribution parameters:
kG1f_0 = DataFrame(name="kG1f", vals=[kG1f_vals], errs=[kG1f_errors], wts=[kG1f_wts], err_types=[kG1f_err_types])
kG1r_0 = DataFrame(name="kG1r", vals=[kG1r_vals], errs=[kG1r_errors], wts=[kG1f_wts], err_types=[kG1f_err_types])
KdG1_0 = DataFrame(name="KdG1", vals=[KdG1_vals], errs=[KdG1_errors], wts=[[12.0, 10.0]], err_types=[[0., 0.]])
append!(expts_df, kG1f_0)
append!(expts_df, kG1r_0)
append!(expts_df, KdG1_0)



## EGF-EGFR binding prior information:
# Forward binding:
kEGFf_vals = [6.3e1 ± 1.9e1] #.* 1e15 * 1e6/6.022e23
kEGFf_errors = [kEGFf_vals[i].err for i in eachindex(kEGFf_vals)] .* sqrt(3.0)

# Reverse binding:
kEGFr_vals = [0.16 ± 0.05]
kEGFr_errors = [i.err for i in kEGFr_vals] .* sqrt(3.0)

# Dissociation constant:
KdEGF_vals = kEGFr_vals./kEGFf_vals
KdEGF_errors = [i.err for i in KdEGF_vals]

# DataFrames for getting prior distributions params:
kEGFf_0 = DataFrame(name="kEGFf", 
    vals=[[i.val for i in kEGFf_vals]],
    errs=[kEGFf_errors], 
    wts=[[14.]], 
    err_types=[zeros(length(kEGFf_vals))]
    )
kEGFr_0 = DataFrame(name="kEGFr", 
    vals=[[i.val for i in kEGFr_vals]],
    errs=[kEGFr_errors], 
    wts=[[14.]], 
    err_types=[zeros(length(kEGFf_vals))]
    )
KdEGF_0 = DataFrame(name="KdEGF", 
    vals=[[i.val for i in KdEGF_vals]], 
    errs=[KdEGF_errors], 
    wts=[[14.]], 
    err_types=[zeros(length(KdEGF_vals))]
    )
append!(expts_df, kEGFf_0)
append!(expts_df, kEGFr_0)
append!(expts_df, KdEGF_0)



## EGFR dimerization prior information:
#= Propagate error in kdr by log-transforming first (to get back to additive error prop.), then exponentiate back:
    (I think this is correct, but not 100% certain) =#
σ_Kdd = sqrt(100)*(1.9-0.068)/3.92
δ_kdr = exp((log(1.2) ± log(1.1) + log(3.8e-1)±log(σ_Kdd)).err).val

# DataFrames for getting prior distributions params:
kdf_0 = DataFrame(name="kdf", vals=[[1.2]], errs=[[1.1]], wts=[[14.]], err_types=[[1.]])
Kdd_0 = DataFrame(name="Kdd", vals=[[3.8e-1]], errs=[[σ_Kdd]], wts=[[14.]], err_types=[[1.]])
kdr_0 = DataFrame(name="kdr", vals=[[1.2*3.8e-1]], errs=[[δ_kdr]], wts=[[14.]], err_types=[[1.]])
append!(expts_df, kdf_0)
append!(expts_df, Kdd_0)
append!(expts_df, kdr_0)



## EGFR phosphorylation:
kp_vals = [14.4 ± 0.5, 17.4 ± 0.6, 7.2 ± 0.3, 12.9 ± 0.4, 13.1 ± 0.4, 15.1 ± 0.2]
kp_errors = [i.err for i in kp_vals] .* sqrt(4.0)

# DataFrames for getting prior distributions params:
kp_0 = DataFrame(name="kp", 
    vals=[[i.val for i in kp_vals]], 
    errs=[kp_errors], 
    wts=[12.0.*ones(length(kp_vals))], 
    err_types=[zeros(length(kp_vals))]
    )
append!(expts_df, kp_0)



## EGFR dephosphorylation prior information:
kdp_vals = [8.0 ± 0.8,
            40.2 ± (2.76.* sqrt(2.0)),
            52.8 ± (9.0.* sqrt(2.0)),
            36 ± (14.0* sqrt(2.0)),
            127.2 ± (37.8.* sqrt(2.0))
            ]
kdp_errors = [i.err for i in kdp_vals] 

# DataFrames for getting prior distributions params:
kdp_0 = DataFrame(name="kdp", 
    vals=[[i.val for i in kdp_vals]], 
    errs=[kdp_errors], 
    wts=[[10., 10., 10., 10., 10.]], 
    err_types=[zeros(length(kdp_vals))]
    )
append!(expts_df, kdp_0)



## GAB1 (de)phosphorylation prior information:
kG1p_0 = DataFrame(name="kG1p", vals=[[0.42]] , errs=[[10.]], wts=[[12.]], err_types=[[1.]])
kG1dp_0 = DataFrame(name="kG1dp", vals=[[9.5]] , errs=[[10.]], wts=[[12.]], err_types=[[1.]])
append!(expts_df, kG1p_0)
append!(expts_df, kG1dp_0)



## SFK (in)activation prior information: Using same as GAB1 (de)phosphorylation
kSa_0 = DataFrame(name="kSa", vals=[[0.42]] , errs=[[10.]], wts=[[12.]], err_types=[[1.]])  # activation
kSi_0 = DataFrame(name="kSi", vals=[[9.5]] , errs=[[10.]], wts=[[12.]], err_types=[[1.]])   # deactivation
append!(expts_df, kSa_0)
append!(expts_df, kSi_0)


## Diffusivity prior information:
for i in axes(D_df,1)
    D_0ᵢ = DataFrame(name=D_df.species[i], 
        vals=[[D_df.D[i].val]], 
        errs=[[D_df.D[i].err]],
        wts=[[12.]],
        err_types=[[0.]]
        )
    append!(expts_df, D_0ᵢ)
end
unique!(expts_df)   # remove duplicates




## =============== Generate prior distribution parameters =============== ##
prior_pars_df = DataFrame()
for i in axes(expts_df,1)
    nameᵢ = expts_df.name[i]
    Pᵢ = [expts_df.vals[i] expts_df.errs[i] expts_df.wts[i] expts_df.err_types[i]]
    μᵢ, σᵢ = log.(calcModeSpread(Pᵢ))
    append!(prior_pars_df, DataFrame(param=nameᵢ, μ=μᵢ, σ=σᵢ))
end


## Get multivariate distribution params for kf/kr/Kd parameter combos:
fin = 0
while fin == 0 
    EG2_dist, G2G1_dist, G1S2_dist, EGF_dist, dim_dist = try
        # -- EGFR-GRB2 binding:
        println()
        println("EGFR-GRB2 binding")
        EG2_dist = @pipe prior_pars_df |> 
            @subset(_, occursin.(r"kG2|KdG2", :param)) |>  
            multivariate3param(_.μ[occursin.(r"Kd", _.param)][1], _.σ[occursin.(r"Kd", _.param)][1], 
                _.μ[occursin.(r"f$", _.param)][1], _.σ[occursin.(r"f$", _.param)][1], 
                _.μ[occursin.(r"r$", _.param)][1], _.σ[occursin.(r"r$", _.param)][1])[1]
        println()
        sleep(0.25)

        # -- GRB2-GAB1:
        println("GRB2-GAB1")
        G2G1_dist = @pipe prior_pars_df |> 
            @subset(_, occursin.(r"kG1|KdG1", :param)) |>  
            multivariate3param(_.μ[occursin.(r"Kd", _.param)][1], _.σ[occursin.(r"Kd", _.param)][1], 
                _.μ[occursin.(r"f$", _.param)][1], _.σ[occursin.(r"f$", _.param)][1], 
                _.μ[occursin.(r"r$", _.param)][1], _.σ[occursin.(r"r$", _.param)][1])[1]
        println()
        sleep(0.25)

        # -- GAB1-SHP2 binding:
        println("GAB1-SHP2 binding")
        G1S2_dist = @pipe prior_pars_df |> 
            @subset(_, occursin.(r"kS2|KdS2", :param)) |>  
            multivariate3param(_.μ[occursin.(r"Kd", _.param)][1], _.σ[occursin.(r"Kd", _.param)][1], 
                _.μ[occursin.(r"f$", _.param)][1], _.σ[occursin.(r"f$", _.param)][1], 
                _.μ[occursin.(r"r$", _.param)][1], _.σ[occursin.(r"r$", _.param)][1])[1]
        println()
        sleep(0.25)

        # -- EGF-EGFR binding:
        println("EGF-EGFR binding:")
        EGF_dist = @pipe prior_pars_df |> 
            @subset(_, occursin.(r"kEGF|KdEGF", :param)) |> 
            multivariate3param(_.μ[occursin.(r"Kd", _.param)][1], _.σ[occursin.(r"Kd", _.param)][1], 
                _.μ[occursin.(r"f$", _.param)][1], _.σ[occursin.(r"f$", _.param)][1], 
                _.μ[occursin.(r"r$", _.param)][1], _.σ[occursin.(r"r$", _.param)][1])[1]
        println()
        sleep(0.25)

        # -- EGFR dimerization:
        println("EGFR dimerization:")
        dim_dist = @pipe prior_pars_df |> 
            @subset(_, occursin.(r"kdf|kdr|Kdd", :param)) |> 
            multivariate3param(_.μ[occursin.(r"Kd", _.param)][1], _.σ[occursin.(r"Kd", _.param)][1], 
                _.μ[occursin.(r"f$", _.param)][1], _.σ[occursin.(r"f$", _.param)][1], 
                _.μ[occursin.(r"r$", _.param)][1], _.σ[occursin.(r"r$", _.param)][1])[1]
        sleep(0.25)
        return EG2_dist, G2G1_dist, G1S2_dist, EGF_dist, dim_dist
    catch
        return nothing, nothing, nothing, nothing, nothing
    end

    if any(isnothing.([EG2_dist, G2G1_dist, G1S2_dist, EGF_dist, dim_dist]))
        global fin = 0
    else
        global EG2_dist, G2G1_dist, G1S2_dist, EGF_dist, dim_dist
        global fin = 1
    end
end



## ===== Collect univariate and multivariate dist parameters for input to Turing model ===== ##
uvpars = @subset(prior_pars_df, occursin.(r"kp|kdp|^D|kSa|kSi|kG1p|kG1dp",:param))
mvpars = (G2=EG2_dist, G1=G2G1_dist, S2=G1S2_dist, EGF=EGF_dist, dim=dim_dist)


# -- Diffusivity modes:
Diffs0 = exp.([uvpars[uvpars.param.=="Dsfk",:].μ[1],
    uvpars[uvpars.param.=="Dg2",:].μ[1],
    uvpars[uvpars.param.=="Dg2g1",:].μ[1],
    uvpars[uvpars.param.=="Dg2g1s2",:].μ[1],
    uvpars[uvpars.param.=="Dg1",:].μ[1],
    uvpars[uvpars.param.=="Dg1s2",:].μ[1],
    uvpars[uvpars.param.=="Ds2",:].μ[1]])

# -- Kinetic parameter modes:
kG2 = exp.(mvpars.G2.normal.μ)   # EGFR-GRB2 binding
kG1 = exp.(mvpars.G1.normal.μ)   # GRB2-GAB1 binding
kS2 = exp.(mvpars.S2.normal.μ)   # GAB1-SHP2 binding
kEGF = exp.(mvpars.EGF.normal.μ)   # EGF-EGFR binding
kdim = exp.(mvpars.dim.normal.μ)   # EGFR dimerization
kG1p = exp.(uvpars[uvpars.param.=="kG1p",:].μ[1])
kG1dp = exp.(uvpars[uvpars.param.=="kG1dp",:].μ[1])
kSa = exp.(uvpars[uvpars.param.=="kSa",:].μ[1])
kSi = exp.(uvpars[uvpars.param.=="kSi",:].μ[1])
kp = exp.(uvpars[uvpars.param.=="kp",:].μ[1])
kdp = exp.(uvpars[uvpars.param.=="kdp",:].μ[1])

kvals0 = [kS2[2]/kS2[1], kS2[2], kG1[1], kG1[2], kG2[2]/kG2[1], kG2[2],
        kG1p, kG1dp, kSa, kSi, kp, kdp, kEGF[2]/kEGF[1],
        kEGF[2], EGF, kdim[2]/kdim[1], kdim[2]]

# -- Combined parameter vector:
pvals0 = [Diffs0; kvals0]

# -- Indices for combined diffusivity and kinetic parameter vector:
D_inds = collect(1:length(Diffs0))
k_inds = collect((D_inds[end]+1):(D_inds[end]+length(kvals0)))