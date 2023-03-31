## =============== Load packages =============== ##
using ModelingToolkit, MethodOfLines, DifferentialEquations, DomainSets, SciMLSensitivity
using NumericalIntegration
# using Symbolics: scalarize


#= If on an Apple silicon Mac (M1, M2, etc), set the number of BLAS threads below to 1 for 
best performance. If on Windows, set the number of BLAS threads to the number of physical 
cores (NOT the number of logical/hyperthreaded cores) on your CPU. 
=#
# using LinearAlgebra
# LinearAlgebra.BLAS.set_num_threads(1)     
# using MKL   # if on an Intel CPU, using Intel's MKL BLAS library can massively improve ODE solver speed

## =============== Load parameter values, distributions, etc. =============== ##
include("get_param_posteriors.jl")

## =============== Define model parameters, variables, and domains =============== ##
# -- Grid/time bounds:
r_min = t_min = 0.0
R = 10.0
dr = 0.2    # spatial step size
tf = 5.0  # final simulation time
volCF = 1/(4/3*pi*R^3) # conversion factor from cell⁻¹ to 1/μm³ units for a spherical cell
surfCF = 1/(4*pi*R^2) # conversion factor from cell⁻¹ to 1/μm² units for a spherical cell

vol = 1/volCF
sa = 1/surfCF


# -- Define initial concentrations in molecules/μm³ or molecules/μm² (for surface species):
CoSFK = 6.0e5 * volCF  # Total concentration of SFK (molec/μm³)
CoG2 = 6.0e5 * volCF   # Total concentration of GRB2 (molec/μm³)
CoG1 = 6.0e5 * volCF   # Total concentration of GAB1 (molec/μm³)
CoS2 = 6.0e5 * volCF   # Total concentration of SHP2 (molec/μm³)
CoEGFR = 6.0e5 * surfCF    # Total concentration of EGFR (molec/μm²)


# -- Combined diffusivity and kinetic parameter values:
D_S, D_G2, D_G2G1, D_G2G1S2, D_G1, D_G1S2, D_S2 = Diffs
kS2f, kS2r, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp, kSa, kSi, kp, kdp, kEGFf, kEGFr, EGF, kdf, kdr = kvals
Co = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations
# D_inds = collect(1:length(Diffs))
# k_inds = collect((D_inds[end]+1):(D_inds[end]+length(kvals)))
# Co_inds = collect((k_inds[end]+1):(k_inds[end]+length(Co)))



## ============== Model equations =============== ##
# -- Define MTK parameters, variables, and differential opertators:
# @parameters r t
@variables r t
ModelingToolkit.@parameters begin
  D1  # = D_S
  D2  # = D_G2
  D3  # = D_G2G1
  D4  # = D_G2G1S2
  D5  # = D_G1
  D6  # = D_G1S2
  D7  # = D_S2
  k1  # = kS2f
  k2  # = kS2r
  k3  # = kG1f
  k4  # = kG1r
  k5  # = kG2f
  k6  # = kG2r
  k7  # = kG1p
  k8  # = kG1dp
  k9  # = kSa
  k10 # = kSi
  k11 # = kp
  k12 # = kdp
  k13 # = kEGFf
  k14 # = kEGFr
  k15 # = EGF
  k16 # = kdf
  k17 # = kdr
  # C01 = CoSFK
  # C02 = CoG2
  # C03 = CoG1
  # C04 = CoS2
  # C05 = CoEGFR
end
# @parameters D[1:length(Diffs)], k[1:length(kvals)], C0[1:length(Co)]
# @parameters D1, D2, D3, D4, D5, D6, D7
# @parameters k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k1₆, k1₇
# @parameters Co1, Co2, Co3, Co4, Co5
@variables iSFK(..) aSFK(..) GAB1(..) PG1(..) GRB2(..) G2G1(..) G2PG1(..) SHP2(..) PG1S(..) G2PG1S(..)
@variables mE(..) mES(..) mESmES(..) E(..) EG2(..) EG2G1(..) EG2PG1(..) EG2PG1S(..)
Dt = Differential(t)    # ∂/∂t or d/dt
Dr = Differential(r)    # ∂/∂r
Drr = Differential(r)^2     # ∂²/∂r²
∇²(u,r) = 2.0/r*Dr(u) + Drr(u)    # spherical Laplacian


eq = [# -- Cytosolic species:
    # Inactive SFKs (iSFK):
      Dt(iSFK(r,t)) ~ D1*∇²(iSFK(r,t), r) + k10*aSFK(r,t),

    # Active SFKs (aSFK):
      Dt(aSFK(r,t)) ~ D1*∇²(aSFK(r,t), r) - k10*aSFK(r,t),

    # GAB1:
      Dt(GAB1(r,t)) ~ D5*∇²(GAB1(r,t),r) +
        -k3*GAB1(r,t)*GRB2(r,t) + k4*G2G1(r,t) +
        -k7*aSFK(r,t)*GAB1(r,t) + k8*PG1(r,t),

    # pGAB1 (PG1):
      Dt(PG1(r,t)) ~ D5*∇²(PG1(r,t), r) + 
        k7*aSFK(r,t)*GAB1(r,t) - k8*PG1(r,t) +
        -k3*PG1(r,t)*GRB2(r,t) + k4*G2PG1(r,t) + 
        -k1*PG1(r,t)*SHP2(r,t) + k2*PG1S(r,t),

    # GRB2:
      Dt(GRB2(r,t)) ~ D2*∇²(GRB2(r,t), r) +
        -k3*GAB1(r,t)*GRB2(r,t) + k4*G2G1(r,t) + 
        -k3*PG1(r,t)*GRB2(r,t) + k4*G2PG1(r,t) +
        -k3*PG1S(r,t)*GRB2(r,t) + k4*G2PG1S(r,t),

    # GRB2-GAB1 (G2G1):
      Dt(G2G1(r,t)) ~ D3*∇²(G2G1(r,t), r) + 
        k3*GAB1(r,t)*GRB2(r,t) - k4*G2G1(r,t) +
        -k7*aSFK(r,t)*G2G1(r,t) + k8*G2PG1(r,t),

    # GRB2-pGAB1 (G2PG1):
      Dt(G2PG1(r,t)) ~ D3*∇²(G2PG1(r,t), r) + 
        k7*aSFK(r,t)*G2G1(r,t) - k8*G2PG1(r,t) +
        k3*PG1(r,t)*GRB2(r,t) - k4*G2PG1(r,t) + 
        -k1*G2PG1(r,t)*SHP2(r,t) + k2*G2PG1S(r,t),

    # SHP2:
      Dt(SHP2(r,t)) ~ D7*∇²(SHP2(r,t), r) +
        -k1*PG1(r,t)*SHP2(r,t) + k2*PG1S(r,t) +
        -k1*G2PG1(r,t)*SHP2(r,t) + k2*G2PG1S(r,t),

    # pGAB1-SHP2 (PG1S):
      Dt(PG1S(r,t)) ~ D6*∇²(PG1S(r,t), r) + 
        k1*PG1(r,t)*SHP2(r,t) - k2*PG1S(r,t) -
        k3*GRB2(r,t)*PG1S(r,t) + k4*G2PG1S(r,t),

    # GRB2-pGAB1-SHP2 (G2PG1S):
      Dt(G2PG1S(r,t)) ~ D4*∇²(G2PG1S(r,t), r) + 
        k1*G2PG1(r,t)*SHP2(r,t) - k2*G2PG1S(r,t) +
        k3*GRB2(r,t)*PG1S(r,t) - k4*G2PG1S(r,t),
      

    # -- Membrane-bound species:
    # Monomeric EGFR (mE):
      Dt(mE(t)) ~ -k13*k15*mE(t) + k14*mES(t),

    # EGF-EGFR (mES):
      Dt(mES(t)) ~ k13*k15*mE(t) - k14*mES(t) - 2.0*k16*mES(t)^2 + 2.0*k17*mESmES(t),

    # EGF-EGFR dimer (mESmES):
      Dt(mESmES(t)) ~ k16*mES(t)^2 - k17*mESmES(t) - k11*mESmES(t) + k12*E(t),

    # pEGFR dimer (E):
      Dt(E(t)) ~ k11*mESmES(t) - k12*E(t) +
        -k5*GRB2(R,t)*E(t) + k6*EG2(t) + 
        -k5*G2G1(R,t)*E(t) + k6*EG2G1(t) +
        -k5*G2PG1(R,t)*E(t) + k6*EG2PG1(t) +
        -k5*G2PG1S(R,t)*E(t) + k6*EG2PG1S(t),

    # pEGFR-GRB2 (EG2):  
      Dt(EG2(t)) ~ k5*GRB2(R,t)*E(t) - k6*EG2(t) +
        -k3*GAB1(R,t)*EG2(t) + k4*EG2G1(t) +
        -k3*PG1(R,t)*EG2(t) + k4*EG2PG1(t) +
        -k3*PG1S(R,t)*EG2(t) + k4*EG2PG1S(t),

    # pEGFR-GRB2-GAB1 (EG2G1):  
      Dt(EG2G1(t)) ~ k3*GAB1(R,t)*EG2(t) - k4*EG2G1(t) + 
        k5*G2G1(R,t)*E(t) - k6*EG2G1(t),

    # pEGFR-GRB2-pGAB1 (EG2PG1):
      Dt(EG2PG1(t)) ~ k3*PG1(R,t)*EG2(t) - k4*EG2PG1(t) + 
        k5*G2PG1(R,t)*E(t) - k6*EG2PG1(t) +
        -k1*SHP2(R,t)*EG2PG1(t) + k2*EG2PG1S(t),

    # pEGFR-GRB2-pGAB1-SHP2 (EG2PG1S):
      Dt(EG2PG1S(t)) ~ k1*SHP2(R,t)*EG2PG1(t) - k2*EG2PG1S(t) +
        k3*PG1S(R,t)*EG2(t) - k4*EG2PG1S(t) +
        k5*G2PG1S(R,t)*E(t) - k6*EG2PG1S(t)    
      ]

       
# -- Initial and boundary conditions:
bcs = [# -- Boundary conditions @ r = 0:
       Dr(iSFK(0,t)) ~ 0.0,
       Dr(aSFK(0,t)) ~ 0.0,
       Dr(GRB2(0,t)) ~ 0.0,
       Dr(GAB1(0,t)) ~ 0.0,
       Dr(PG1(0,t)) ~ 0.0,
       Dr(SHP2(0,t)) ~ 0.0,
       Dr(G2G1(0,t)) ~ 0.0,
       Dr(G2PG1(0,t)) ~ 0.0,
       Dr(PG1S(0,t)) ~ 0.0,
       Dr(G2PG1S(0,t)) ~ 0.0,

       # -- Boundary conditions @ r = R:
       D1*Dr(iSFK(R,t)) ~ -k9*iSFK(R,t)*2.0*(E(t)+EG2(t)+EG2G1(t)+EG2PG1(t)+EG2PG1S(t)),
       D1*Dr(aSFK(R,t)) ~ k9*iSFK(R,t)*2.0*(E(t)+EG2(t)+EG2G1(t)+EG2PG1(t)+EG2PG1S(t)),
       D5*Dr(GAB1(R,t)) ~ k4*EG2G1(t) - k3*GAB1(R,t)*EG2(t),
       D5*Dr(PG1(R,t)) ~ k4*EG2PG1(t) - k3*PG1(R,t)*EG2(t),
       D2*Dr(GRB2(R,t)) ~ k6*EG2(t) - k5*GRB2(R,t)*E(t),
       D3*Dr(G2G1(R,t)) ~ k6*EG2G1(t) - k5*G2G1(R,t)*E(t),
       D3*Dr(G2PG1(R,t)) ~ k6*EG2PG1(t) - k5*G2PG1(R,t)*E(t),
       D7*Dr(SHP2(R,t)) ~ k2*EG2PG1S(t) - k1*SHP2(R,t)*EG2PG1(t),
       D6*Dr(PG1S(R,t)) ~ k4*EG2PG1S(t) - k3*PG1S(R,t)*EG2(t),
       D4*Dr(G2PG1S(R,t)) ~ k6*EG2PG1S(t) - k5*G2PG1S(R,t)*E(t),
       
       # ===== Initial conditions ===== #
       # -- Cytosolic species:
       iSFK(r,0) ~ CoSFK,#C01,
       aSFK(r,0) ~ 0.0,
       GRB2(r,0) ~ CoG2, #C02,
       GAB1(r,0) ~ CoG1, #C03,
       PG1(r,0) ~ 0.0,
       SHP2(r,0) ~ CoS2, #C04,
       G2G1(r,0) ~ 0.0,
       G2PG1(r,0) ~ 0.0,
       PG1S(r,0) ~ 0.0,
       G2PG1S(r,0) ~ 0.0,

       # -- Membrane-bound species:
       mE(0) ~ CoEGFR, #C05,
       mES(0) ~ 0.0,
       mESmES(0) ~ 0.0,
       E(0) ~ 0.0,
       EG2(0) ~ 0.0,
       EG2G1(0) ~ 0.0,
       EG2PG1(0) ~ 0.0,
       EG2PG1S(0) ~ 0.0,
       ] 

# -- Define space and time domains:
domains = [r ∈ Interval(r_min, R),
           t ∈ Interval(t_min, tf)]
       

## Define PDESystem:
# -- Model variables:
vars = [iSFK(r,t), aSFK(r,t), GAB1(r,t), PG1(r,t), GRB2(r,t), G2G1(r,t), G2PG1(r,t), SHP2(r,t), PG1S(r,t), G2PG1S(r,t),
    mE(t), mES(t), mESmES(t), E(t), EG2(t), EG2G1(t), EG2PG1(t), EG2PG1S(t)
    ]
# -- Model parameters:
pars = [D1=>D_S, D2=>D_G2, D3=>D_G2G1, D4=>D_G2G1S2, D5=>D_G1, D6=>D_G1S2, D7=>D_S2,
    k1=>kS2f, k2=>kS2r, k3=>kG1f, k4=>kG1r, k5=>kG2f, k6=>kG2r, k7=>kG1p, k8=>kG1dp,
    k9=>kSa, k10=>kSi, k11=>kp, k12=>kdp, k13=>kEGFf, k14=>kEGFr, k15=>EGF, k16=>kdf, k17=>kdr
    # C01=>CoSFK, C02=>CoG2, C03=>CoG1, C04=>CoS2, C05=>CoEGFR
    ]
# pars = [D[1]=>D_S, D[2]=>D_G2, D[3]=>D_G2G1, D[4]=>D_G2G1S2, D[5]=>D_G1, D[6]=>D_G1S2, D[7]=>D_S2,
#     k[1]=>kS2f, k[2]=>kS2r, k[3]=>kG1f, k[4]=>kG1r, k[5]=>kG2f, k[6]=>kG2r, k[7]=>kG1p, k[8]=>kG1dp,
#     k[9]=>kSa, k[10]=>kSi, k[11]=>kp, k[12]=>kdp, k[13]=>kEGFf, k[14]=>kEGFr, k[15]=>EGF, k[16]=>kdf, k[17]=>kdr, 
#     C0[1]=>CoSFK, C0[2]=>CoG2, C0[3]=>CoG1, C0[4]=>CoS2, C0[5]=>CoEGFR
#     ]
# pars2 = [D_S, D_G2, D_G2G1, D_G2G1S2, D_G1, D_G1S2,D_S2,
#     kS2f, kS2r, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp,
#     kSa, kSi, kp, kdp, kEGFf, kEGFr, EGF, kdf, kdr, 
#     CoSFK, CoG2, CoG1, CoS2, CoEGFR
#     ]

# -- Define model:
@named pdesys = PDESystem(eq,bcs,domains,[r,t], vars, pars)
# @named pdesys = PDESystem(eq,bcs,domains,[r,t], vars, pvals)


## Method of lines discretization:
discretization = MOLFiniteDifference([r=>dr],t)

# -- Discretize the PDE system: Note that this may take several minutes depending on the spatial step size!
@time prob = discretize(pdesys, discretization, jac=true)
# @time prob = discretize(pdesys, discretization)


## Check sparsity pattern of the Jacobian:
# using Symbolics
# du0 = copy(prob.u0)
# @time jac_sparsity = Symbolics.jacobian_sparsity((du,u)->prob.f(du,u,pvals,0.0), du0, prob.u0)
# display(jac_sparsity)
# 1-sum(iszero.(jac_sparsity))/prod(size(jac_sparsity))



## Solve the problem:
# dt_save = 0.25
# dt_save = 5.0
# prob2 = remake(prob, p=pars2)
@time testsol = solve(prob, QNDF(), reltol=1e-4, saveat=0.25)
# @time sol = solve(prob2, QNDF(), reltol=1e-4, saveat=dt_save)
# # @time sol = solve(prob, QNDF(), reltol=1e-4, wrap=Val(false), saveat=dt_save)
# # @time sol = solve(prob, QNDF(); wrap=Val(false))
# # @time sol = solve(prob2, QNDF())


## Ensemble simulations method for MoL version of the model:
"""
    run_ensemble(prob::DEProblem, ensemble, Co; 
        tf = 5.0, 
        abstol = 1e-6,
        reltol = 1e-4,
        D_inds = 1:7, 
        k_inds = 8:24
        )\n
Function for ensemble simulations with the GAB1-SHP2 reaction-diffusion model.\n
...
Function for quickly running model simulations with parameter ensembles. Returns a DataFrame 
containing the full model solutions for each parameter set. By default the simulations 
are multithreaded.

# Positional arguments
prob = DEProblem to solve (problem generated from discretized PDESystem)\n
r = spatial independent variable\n
t = time variable\n
ensemble = matrix containing parameter sets to use for model ensemble simulations\n
Co = vector of initial species concentrations\n
# Keyword arguments
tf = final simulation time\n
absolt = absolute tolerance for ODE solver\n
reltol = relative tolerance for ODE solver\n
odealg = ODE solver algorithm (defaults to QNDF)\n
"""
function run_ensemble(prob::DEProblem, ensemble; 
    tf=5.0,
    abstol=1e-6, reltol=1e-4,
    odealg=QNDF(),
    )
    retro_full_df = DataFrame() # for saving full model solutions

    prog = Progress(size(ensemble,1))     # initialize progress meter
    lk = ReentrantLock()    # lock for multithreading

    Threads.@threads for j in axes(ensemble,1)
    # for j in axes(ensemble,1)
        # -- Set parameter values:
        pvalsⱼ = ensemble[j,:]
        
        # -- Run model:
        prob2 = remake(prob, p=pvalsⱼ, tspan=(0., tf))
        solⱼ = solve(prob2, odealg; abstol=abstol, reltol=reltol, verbose=false)

        if solⱼ.retcode == :Success
            Threads.lock(lk) do
                # -- Save results:
                df_fullⱼ = DataFrame(sol=[solⱼ], index=j)
                append!(retro_full_df, df_fullⱼ)  # save full model results
            end
        else
        end
        next!(prog) # update progress meter
    end
    return retro_full_df
end