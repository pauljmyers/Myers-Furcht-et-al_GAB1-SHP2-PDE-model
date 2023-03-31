## Required packages to run the functions below:
@everywhere using NumericalIntegration, Distributed, ProgressMeter, SharedArrays

# -- Run scripts to get necessary parameter values, etc.:
include("get_param_posteriors.jl")


## ============== Define model parameters ============== ##
@everywhere begin
    # -- Spatiotemporal grid parameters:
    dr = 0.2 # length of each spatial step
    R = 10.0 #  Radius of cell
    Nts = 100 #  Number of time points to store
    tf = 5.0 # final integration time
    tvec = collect(0:tf/Nts:tf) # time vector
    volCF = 1/(4/3*pi*R^3) # conversion factor from cell⁻¹ to 1/μm³ units for a spherical cell
    surfCF = 1/(4*pi*R^2) # conversion factor from cell⁻¹ to 1/μm² units for a spherical cell

    # -- Define initial concentrations in molecules/μm³ or molecules/μm² (for surface species):
    CoSFK = 6.0e5*volCF  # Total concentration of SFK (molec/μm³)
    CoG2 = 6.0e5*volCF   # Total concentration of GRB2 (molec/μm³)
    CoG1 = 6.0e5*volCF   # Total concentration of GAB1 (molec/μm³)
    CoS2 = 6.0e5*volCF   # Total concentration of SHP2 (molec/μm³)
    CoEGFR = 6.0e5*surfCF    # Total concentration of EGFR (molec/μm²)
    Co = [CoSFK, CoG2, CoG1, CoS2, CoEGFR]     # Vector of initial concentrations
end


## Functions:
@everywhere begin
    """
        sapdesolver_membSFK(Co::Vector{Float64}, D::Vector{Float64}, k::Vector{Float64}; 
            R::Float64=10.0, dr::Float64=0.1, tf::Float64=5.0,
            dt::Float64=1.0/(2.0*(maximum(D)/dr.^2 + sum(k)/4))*0.99,
            maxiters=20,
            tol::Float64 = 1.0e-4)\n
    GAB1-SHP2 reaction-diffusion model solver function, modified to only provide outputs 
    at the final time point of the simulation for speed when performing sensitivity analysis (which only needs
    outputs at the final time point).\n
    ...
    # Positional arguments
    Co = vector of initial species concentrations in the following order: SFKs, GRB2, GAB1, SHP2, and EGFR.
    D = vector of diffusivities for cytosolic species in the following order: SFKs, GRB2, GRB2-GAB1, GRB2-GAB1-SHP2, GAB1, GAB1-SHP2, SHP2
    k = vector of model parameters and kinetic rate constants for model reactions in the following
    order: kS2f, kS2r, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp, kSa, kSi, kp, kdp,
    kEGFf, kEGFr, [EGF], kdf, kdr.\n
    # Keyword arguments
    R = simulation radius (radius of the cell)\n
    dr = spatial step size\n
    tf = final simulation time\n
    dt = time step size\n
    maxiters = maximum number of semi-implicit loop iterations\n
    tol = error tolerance for semi-implicit finite difference scheme at cell membrane.
    """
    function sapdesolver_membSFK(Co::Vector{Float64}, D::Vector{Float64}, k::Vector{Float64}; 
        R::Float64=R, dr::Float64=dr, tf::Float64=tf,
        dt::Float64=1.0/(2.0*(maximum(D)/dr.^2 + sum(k)/4))*0.99,
        maxiters=20,
        tol::Float64 = 1.0e-3)
        ## DIFFUSIVITIES:
        D_S = D[1]     # inactive SFKs
        D_Sa = 1e-32    # actuve SFKs
        D_G2 = D[2]    # GRB2
        D_G2G1 = D[3]  # GRB2-GAB1
        D_G2G1S2 = D[4]    # GRB2-GAB1-SHP2
        D_G1 = D[5]    # GAB1
        D_G1S2 = D[6]  # GAB1-SHP2
        D_S2 = D[7]    # SHP2

        ## KINETIC RATE CONSTANTS:
        kS2f = k[1]     # Rate constant for SHP2 binding to pGAB1, μm³/(molec*min)
        kS2r = k[2]     # Rate constant for SHP2 unbinding from pGAB1, 1/min
        kG1f = k[3]    # Rate constant for GAB1 binding to GRB2, μm³/(molec*min)
        kG1r = k[4]    # Rate constant GAB1 unbinding from GRB2, 1/min
        kG2f = k[5]    # Rate constant for GRB2 binding to EGFR, μm³/(molec*min)
        kG2r = k[6]    # Rate constant for GRB2 unbinding from EGFR, 1/min
        kG1p = k[7]    # GAB1 phosphorylation rate constant, μm³/(molec*min)
        kG1dp = k[8]   # GAB1 dephosphorylation rate constant, min-1
        kSa = k[9]     # SRC activation rate constant μm³/(molec*min)
        kSi = k[10]    # SRC inactivation rate constant, 1/min
        kp = k[11]     # EGFR phosphorylation rate constant, μm²/(molec*min)
        kdp = k[12]    # EGFR dephosphorylation rate constant, 1/min
        kEGFf = k[13]  # EGF binding rate constant, 1/(μM*min)
        kEGFr = k[14]  # EGF unbinding rate constant, 1/min
        EGF = k[15]    # EGF concentration, μM
        kdf = k[16]    # Dimer on rate constant, μm²/(molec*min)
        kdr = k[17]    # Dimer off rate constant, 1/min

        ## SOLVER OPTIONS:
        Nr = Int64(ceil(R/dr))    # Number of spatial steps
        Nt = Int64(ceil(tf/dt))   # Choose the number of time steps
        r = collect(0.0:dr:R)       # Distance grid


        ## SOLVER SETUP:
        # Define initial concentrations in molecules/um^3 (cytoplasmic species)
        # or molecules/um^2 (membrane-bound species):
        CoSFK, CoG2, CoG1, CoS2, CoEGFR = Co
            
        # Initialize solution matrices and vectors:
        iSFK = zeros(length(r), 2)
        aSFK = zeros(length(r), 2)
        GAB1 = zeros(length(r), 2)
        pGAB1 = zeros(length(r), 2)
        GRB2 = zeros(length(r), 2)
        G2G1 = zeros(length(r), 2)
        G2PG1 = zeros(length(r), 2)
        SHP2 = zeros(length(r), 2)
        PG1S = zeros(length(r), 2)
        G2PG1S = zeros(length(r), 2)

        mE = zeros(1,2)
        mES = zeros(1,2)
        mESmES = zeros(1,2)
        E = zeros(1,2)
        EG2 = zeros(1,2)
        EG2G1 = zeros(1,2)
        EG2PG1 = zeros(1,2)
        EG2PG1S = zeros(1,2)

        ## SOLVER
        # Set initial conditions of model species:
        iSFK[:,1] .= CoSFK 
        GAB1[:,1] .= CoG1
        GRB2[:,1] .= CoG2
        SHP2[:,1] .= CoS2
        mE[1] = CoEGFR

        for i in 2:Nt+1 
            for j in 2:Nr # loop over spatial points
                iSFK[j,2] = (D_S*(1/(r[j]*dr)*(iSFK[j+1,1]-iSFK[j-1,1])+(iSFK[j+1,1]-2.0*iSFK[j,1]+iSFK[j-1,1])/dr^2) + 
                    kSi*aSFK[j,1])*dt + iSFK[j,1]

                aSFK[j,2] = (D_Sa*(1/(r[j]*dr)*(aSFK[j+1,1]-aSFK[j-1,1])+(aSFK[j+1,1]-2.0*aSFK[j,1]+aSFK[j-1,1])/dr^2) - 
                    kSi*aSFK[j,1])*dt + aSFK[j,1]

                GAB1[j,2] = (D_G1*(1/(r[j]*dr)*(GAB1[j+1,1]-GAB1[j-1,1])+(GAB1[j+1,1]-2.0*GAB1[j,1]+GAB1[j-1,1])/dr^2) - 
                    kG1f*GAB1[j,1]*GRB2[j,1] + kG1r*G2G1[j,1] - kG1p*aSFK[j,1]*GAB1[j,1] + kG1dp*pGAB1[j,1])*dt + GAB1[j,1]

                pGAB1[j,2]= (D_G1*(1/(r[j]*dr)*(pGAB1[j+1,1]-pGAB1[j-1,1])+(pGAB1[j+1,1]-2.0*pGAB1[j,1]+pGAB1[j-1,1])/dr^2) - 
                    kG1f*pGAB1[j,1]*GRB2[j,1] + kG1r*G2PG1[j,1] + kG1p*aSFK[j,1]*GAB1[j,1] - kG1dp*pGAB1[j,1] - kS2f*SHP2[j,1]*pGAB1[j,1] + kS2r*PG1S[j,1])*dt + pGAB1[j,1]
                    
                GRB2[j,2] = (D_G2*(1/(r[j]*dr)*(GRB2[j+1,1]-GRB2[j-1,1])+(GRB2[j+1,1]-2.0*GRB2[j,1]+GRB2[j-1,1])/dr^2) - 
                    kG1f*GAB1[j,1]*GRB2[j,1] + kG1r*G2G1[j,1] - kG1f*pGAB1[j,1]*GRB2[j,1] + kG1r*G2PG1[j,1] - kG1f*GRB2[j,1]*PG1S[j,1] + kG1r*G2PG1S[j,1])*dt + GRB2[j,1]

                G2G1[j,2] = (D_G2G1*(1/(r[j]*dr)*(G2G1[j+1,1]-G2G1[j-1,1])+(G2G1[j+1,1]-2.0*G2G1[j,1]+G2G1[j-1,1])/dr^2) + 
                    kG1f*GAB1[j,1]*GRB2[j,1] - kG1r*G2G1[j,1] - kG1p*aSFK[j,1]*G2G1[j,1] + kG1dp*G2PG1[j,1])*dt + G2G1[j,1]

                G2PG1[j,2]= (D_G2G1*(1/(r[j]*dr)*(G2PG1[j+1,1]-G2PG1[j-1,1])+(G2PG1[j+1,1]-2.0*G2PG1[j,1]+G2PG1[j-1,1])/dr^2) + 
                    kG1f*pGAB1[j,1]*GRB2[j,1] - kG1r*G2PG1[j,1] + kG1p*aSFK[j,1]*G2G1[j,1] - kG1dp*G2PG1[j,1] - kS2f*SHP2[j,1]*G2PG1[j,1] + kS2r*G2PG1S[j,1])*dt + G2PG1[j,1]

                SHP2[j,2] = (D_S2*(1/(r[j]*dr)*(SHP2[j+1,1]-SHP2[j-1,1])+(SHP2[j+1,1]-2.0*SHP2[j,1]+SHP2[j-1,1])/dr^2) - 
                    kS2f*SHP2[j,1]*pGAB1[j,1] + kS2r*PG1S[j,1] - kS2f*SHP2[j,1]*G2PG1[j,1] + kS2r*G2PG1S[j,1])*dt + SHP2[j,1]

                PG1S[j,2] = (D_G1S2*(1/(r[j]*dr)*(PG1S[j+1,1]-PG1S[j-1,1])+(PG1S[j+1,1]-2.0*PG1S[j,1]+PG1S[j-1,1])/dr^2) + 
                    kS2f*SHP2[j,1]*pGAB1[j,1] - kS2r*PG1S[j,1] - kG1f*GRB2[j,1]*PG1S[j,1] + kG1r*G2PG1S[j,1])*dt + PG1S[j,1]

                G2PG1S[j,2]=(D_G2G1S2*(1/(r[j]*dr)*(G2PG1S[j+1,1]-G2PG1S[j-1,1])+(G2PG1S[j+1,1]-2.0*G2PG1S[j,1]+G2PG1S[j-1,1])/dr^2) + 
                    kG1f*GRB2[j,1]*PG1S[j,1] - kG1r*G2PG1S[j,1] + kS2f*SHP2[j,1]*G2PG1[j,1] - kS2r*G2PG1S[j,1])*dt + G2PG1S[j,1]
            end

            # Enforce zero flux at r = 0:
            iSFK[1,2] = iSFK[2,2]
            aSFK[1,2] = aSFK[2,2]
            GAB1[1,2] = GAB1[2,2]
            pGAB1[1,2] = pGAB1[2,2]
            GRB2[1,2] = GRB2[2,2]
            G2G1[1,2] = G2G1[2,2]
            G2PG1[1,2] = G2PG1[2,2]
            SHP2[1,2] = SHP2[2,2]
            PG1S[1,2] = PG1S[2,2]
            G2PG1S[1,2] = G2PG1S[2,2]

            ## Enforce reactive flux at r = R:
            error = tol*2. # Initialize error with a value > tol
            # for i in 1:maxiters # Stay in semi-implicit loop until error tolerance is met
            while error > tol # Stay in semi-implicit loop until error tolerance is met
                # Store initial guesses for cytosolic species at membrane and membrane species at new time point:
                cyto_old = [iSFK[Nr+1,2], aSFK[Nr+1,2],
                    GAB1[Nr+1,2], pGAB1[Nr+1,2], 
                    GRB2[Nr+1,2], G2G1[Nr+1,2], G2PG1[Nr+1,2], 
                    SHP2[Nr+1,2], PG1S[Nr+1,2], G2PG1S[Nr+1,2]]
                memb_old = [mE[2], mES[2], mESmES[2], E[2], EG2[2], EG2G1[2], EG2PG1[2], EG2PG1S[2]]
                
                Etot= 2.0*(E[2]+EG2[2]+EG2G1[2]+EG2PG1[2]+EG2PG1S[2])
                iSFK[Nr+1,2] = iSFK[Nr,2]/(1+kSa*Etot*dr/D_S)
                aSFK[Nr+1,2] = aSFK[Nr,2]+kSa*iSFK[Nr+1,2]*Etot*dr/D_Sa
                GAB1[Nr+1,2] = (kG1r*EG2G1[2]*dr/D_G1+GAB1[Nr,2])/(1+kG1f*EG2[2]*dr/D_G1)
                pGAB1[Nr+1,2] = (kG1r*EG2PG1[2]*dr/D_G1+pGAB1[Nr,2])/(1+kG1f*EG2[2]*dr/D_G1)
                GRB2[Nr+1,2] = (kG2r*EG2[2]*dr/D_G2+GRB2[Nr,2])/(1+kG2f*E[2]*dr/D_G2)
                G2G1[Nr+1,2] = (kG2r*EG2G1[2]*dr/D_G2G1+G2G1[Nr,2])/(1+kG2f*E[2]*dr/D_G2G1)
                G2PG1[Nr+1,2] = (kG2r*EG2PG1[2]*dr/D_G2G1+G2PG1[Nr,2])/(1+kG2f*E[2]*dr/D_G2G1)
                SHP2[Nr+1,2] = (kS2r*EG2PG1S[2]*dr/D_S2+SHP2[Nr,2])/(1+kS2f*EG2PG1[2]*dr/D_S2)
                PG1S[Nr+1,2] = (kG1r*EG2PG1S[2]*dr/D_G1S2+PG1S[Nr,2])/(1+kG1f*EG2[2]*dr/D_G1S2)
                G2PG1S[Nr+1,2] = (kG2r*EG2PG1S[2]*dr/D_G2G1S2+G2PG1S[Nr,2])/(1+kG2f*E[2]*dr/D_G2G1S2)
                
                
                # Semi-implicit Euler
                # Solve for concentrations of surface species:
                mE[2] = (-kEGFf*EGF*mE[1] + kEGFr*mES[1])*dt + mE[1]
                mES[2] = (kEGFf*EGF*mE[1] - kEGFr*mES[1] - 2*kdf*mES[1]*mES[1] + 2*kdr*mESmES[1])*dt + mES[1]
                mESmES[2] = (kdf*mES[1]*mES[1] - kdr*mESmES[1] - kp*mESmES[1] + kdp*E[1])*dt + mESmES[1]
                E[2] = (kp*mESmES[1] - kdp*E[1] - kG2f*E[1]*GRB2[Nr+1,2] + kG2r*EG2[1] - kG2f*E[1]*G2G1[Nr+1,2] + kG2r*EG2G1[1] - 
                    kG2f*E[1]*G2PG1[Nr+1,2] + kG2r*EG2PG1[1] - kG2f*E[1]*G2PG1S[Nr+1,2] + kG2r*EG2PG1S[1])*dt + E[1]
                EG2[2] = (kG2f*GRB2[Nr+1,2]*E[1] - kG2r*EG2[1] - kG1f*GAB1[Nr+1,2]*EG2[1] + kG1r*EG2G1[1] - kG1f*pGAB1[Nr+1,2]*EG2[1] + 
                    kG1r*EG2PG1[1] - kG1f*PG1S[Nr+1,2]*EG2[1] + kG1r*EG2PG1S[1])*dt + EG2[1]
                EG2G1[2] = (kG2f*G2G1[Nr+1,2]*E[1] - kG2r*EG2G1[1] + kG1f*GAB1[Nr+1,2]*EG2[1] - kG1r*EG2G1[1])*dt + EG2G1[1]
                EG2PG1[2] = (kG2f*G2PG1[Nr+1,2]*E[1] - kG2r*EG2PG1[1] + kG1f*pGAB1[Nr+1,2]*EG2[1] - kG1r*EG2PG1[1] - kS2f*SHP2[Nr+1,2]*EG2PG1[1] +
                    kS2r*EG2PG1S[1])*dt + EG2PG1[1]
                EG2PG1S[2] = (kS2f*SHP2[Nr+1,2]*EG2PG1[1] - kS2r*EG2PG1S[1] + kG1f*PG1S[Nr+1,2]*EG2[1] - kG1r*EG2PG1S[1] + kG2f*G2PG1S[Nr+1,2]*E[1] - 
                    kG2r*EG2PG1S[1])*dt + EG2PG1S[1]
                
                # Store new estimates for cytosolic species at membrane and membrane species at new time point:
                cyto_new = [iSFK[Nr+1,2], aSFK[Nr+1,2], GAB1[Nr+1,2], pGAB1[Nr+1,2], GRB2[Nr+1,2], G2G1[Nr+1,2], G2PG1[Nr+1,2], SHP2[Nr+1,2], PG1S[Nr+1,2], G2PG1S[Nr+1,2]]
                memb_new = [mE[2], mES[2], mESmES[2], E[2], EG2[2], EG2G1[2], EG2PG1[2], EG2PG1S[2]]
                
                # Calculate error between new and old estimates for membrane concentrations at new time point:
                error = maximum([abs.(1. .- cyto_new./cyto_old); abs.(1. .- memb_new./memb_old)])
                # if error <= tol 
                #     break
                # end
            end           
            
            # Update solutions for next iteration:
            iSFK[:,1] .= iSFK[:,2]
            aSFK[:,1] .= aSFK[:,2]
            GAB1[:,1] .= GAB1[:,2]
            pGAB1[:,1] .= pGAB1[:,2]
            GRB2[:,1] .= GRB2[:,2]
            G2G1[:,1] .= G2G1[:,2]
            G2PG1[:,1] .= G2PG1[:,2]
            SHP2[:,1] .= SHP2[:,2]
            PG1S[:,1] .= PG1S[:,2]
            G2PG1S[:,1] .= G2PG1S[:,2]
            mE[1] = mE[2]
            mES[1] = mES[2]
            mESmES[1] = mESmES[2]
            E[1] = E[2]
            EG2[1] = EG2[2]
            EG2G1[1] = EG2G1[2]
            EG2PG1[1] = EG2PG1[2]
            EG2PG1S[1] = EG2PG1S[2]
        end

        # Output solutions:
        iSFK_out = iSFK[:,end]
        aSFK_out = aSFK[:,end]
        # GRB2_out = GRB2[:,end]
        # GAB1_out = GAB1[:,end]
        # SHP2_out = SHP2[:,end]
        # G2G1_out = G2G1[:,end]
        # G2PG1_out = G2PG1[:,end]
        # G2PG1S_out = G2PG1S[:,end]
        # PG1_out = pGAB1[:,end]
        # PG1S_out = PG1S[:,end]
        PG1tot_out = G2PG1[:,end] .+ pGAB1[:,end] .+ PG1S[:,end] .+ G2PG1S[:,end]
        PG1Stot_out = PG1S[:,end] .+ G2PG1S[:,end]

        # EGFR_SHP2_out = EG2PG1S[2]*100.0/CoEGFR
        # mE_out = mE[2]
        # mES_out = mES[2]
        # mESmES_out = mESmES[2]
        # E_out = E[2]
        # pE_out = (2.0*(E[2]+EG2[2]+EG2G1[2]+EG2PG1[2]+EG2PG1S[2]))*100.0/CoEGFR
        # EG2_out = EG2[2]
        # EG2G1_out = EG2G1[2]
        # EG2PG1_out = EG2PG1[2]
        # EG2PG1S_out = EG2PG1S[2]

        ## STORE SOLUTIONS FOR OUTPUT:
        sol = (iSFK = iSFK_out, aSFK = aSFK_out, 
            # GRB2 = GRB2_out, GAB1 = GAB1_out, SHP2 = SHP2_out, G2G1 = G2G1_out,
            # G2PG1 = G2PG1_out, G2PG1S = G2PG1S_out, PG1 = PG1_out, PG1S = PG1S_out, 
            PG1tot = PG1tot_out, PG1Stot = PG1Stot_out 
            # EGFR_SHP2 = EGFR_SHP2_out, pE = pE_out, mE = mE_out, 
            # mES = mES_out, mESmES = mESmES_out,
            # E = E_out, EG2 = EG2_out, EG2G1 = EG2G1_out,
            # EG2PG1 = EG2PG1_out, EG2PG1S = EG2PG1S_out
            )
        return sol, r
    end






    ## GSA function, using Distributed.jl-based parallelism:
    function pmap_fun_allpars(p; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        lenCo = length(Co)
        lenDiffs = length(D)
        lenkvals = length(kvals)
        Co_inds = 1:lenCo
        Diffs_inds = (Co_inds[end]+1):(Co_inds[end] + lenDiffs)
        kvals_inds = (Diffs_inds[end]+1):(Diffs_inds[end] + lenkvals)

        
        Co_temp = p[Co_inds]
        Diffs_temp = p[Diffs_inds] 
        kvals_temp = p[kvals_inds]

        sol, r_sol = sapdesolver_membSFK(Co_temp, Diffs_temp, kvals_temp; R=R, dr=dr, tf=tf) # run model
        aSFK = sol.aSFK
        PG1Stot = sol.PG1Stot

        # Length scale calculations:
        r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
        r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
        r12_pg1s = R - (r_sol[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
        r110_pg1s = R - (r_sol[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

        # Ratio of center-to-surface [PG1S]tot:
        cs_ratio = PG1Stot[1]/PG1Stot[end]

        # Average [PG1S]tot:
        PG1Save = NumericalIntegration.integrate(r_sol, PG1Stot.*r_sol.^2).*3.0./R^3.0
        PG1Save_out = PG1Save[1]

        return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1Save_out]
    end

    function fbatch(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        p_in = [p_batch[:,i] for i in axes(p_batch,2)]
        pmap_out = @showprogress pmap(pmap_fun_allpars, p_in)
        out = hcat(pmap_out...)
        return out
    end


    ## GSA function, using Distributed.jl-based parallelism -- diffusivities & kinetic params only!:
    function pmap_fun_dk(p; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf, maxiters=100)
        lenDiffs = length(D)
        lenkvals = length(kvals)
        Diffs_inds = 1:lenDiffs
        kvals_inds = (Diffs_inds[end]+1):(Diffs_inds[end] + lenkvals)

        Diffs_temp = p[Diffs_inds] 
        kvals_temp = p[kvals_inds]

        sol, r_sol = sapdesolver_membSFK(Co, Diffs_temp, kvals_temp; R=R, dr=dr, tf=tf, tol=1e-3, maxiters=maxiters) # run model
        aSFK = sol.aSFK
        PG1Stot = sol.PG1Stot

        # Length scale calculations:
        r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
        r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
        r12_pg1s = R - (r_sol[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
        r110_pg1s = R - (r_sol[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

        # Ratio of center-to-surface [PG1S]tot:
        cs_ratio = PG1Stot[1]/PG1Stot[end]

        # Average [PG1S]tot:
        PG1Save = NumericalIntegration.integrate(r_sol, PG1Stot.*r_sol.^2).*3.0./R^3.0
        PG1Save_out = PG1Save[1]

        return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1Save_out]
    end

    function fbatch_dk(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        p_in = [exp.(p_batch[:,i]) for i in axes(p_batch,2)]
        # p_in = [10.0 .^ (p_batch[:,i]) for i in axes(p_batch,2)]
        # p_in = [p_batch[:,i] for i in axes(p_batch,2)]
        pmap_out = @showprogress pmap(x->pmap_fun_dk(x; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf), p_in)
        out = hcat(pmap_out...)
        return out
    end

    function fbatch_dk_mt(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf, maxiters=20)
        # @show size(p_batch)
        p_batch = exp.(p_batch)
        # p_batch = 10.0 .^ (p_batch)
        out = zeros(numout, size(p_batch,2))
        prog = Progress(size(out,2))
        Threads.@threads for i in axes(out,2)
            out[:,i] = try
                pmap_fun_dk(p_batch[:,i]; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf, maxiters=maxiters)
            catch
                zeros(size(out[:,i]))
            end
            # @show out[:,i]
            next!(prog)
        end
        return out
    end


    ## All diffusivities varied by the same factor:
    function pmap_fun_dk_combD(p; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        lenkvals = length(kvals)
        
        Diffs_inds = 1
        kvals_inds = (Diffs_inds[end]+1):(Diffs_inds[end] + lenkvals)

        Diffs_multᵢ = p[1]/D[1]   # use multiplier for first diffusivity from its base value to vary all other diffusivities
        Diffs_temp = D.*Diffs_multᵢ     # modify all model sensitivities by same multiplier
        kvals_temp = p[kvals_inds]



        sol, r_sol = sapdesolver_membSFK(Co, Diffs_temp, kvals_temp; R=R, dr=dr, tf=tf) # run model
        aSFK = sol.aSFK
        PG1Stot = sol.PG1Stot

        # Length scale calculations:
        r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
        r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
        r12_pg1s = R - (r_sol[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
        r110_pg1s = R - (r_sol[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

        # Ratio of center-to-surface [PG1S]tot:
        cs_ratio = PG1Stot[1]/PG1Stot[end]

        # Average [PG1S]tot:
        PG1Save =NumericalIntegration.integrate(r_sol, PG1Stot.*r_sol.^2).*3.0./R^3.0
        PG1Save_out = PG1Save[1]

        return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1Save_out]
    end

    function fbatch_dk_combD(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        p_in = [p_batch[:,i] for i in axes(p_batch,2)]
        pmap_out = @showprogress pmap(pmap_fun_dk_combD, p_in)
        out = hcat(pmap_out...)
        return out
    end


    ## GSA function for varying protein concentrations:
    function pmap_fun_concs(p; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf, tol=1e-3, maxiters=20)
        sol, r_sol = sapdesolver_membSFK(p, D, kvals; R=R, dr=dr, tf=tf, tol=tol, maxiters=maxiters) # run model
        aSFK = sol.aSFK
        PG1Stot = sol.PG1Stot

        # Length scale calculations:
        r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
        r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
        r12_pg1s = R - (r_sol[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
        r110_pg1s = R - (r_sol[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

        # Ratio of center-to-surface [PG1S]tot:
        cs_ratio = PG1Stot[1]/PG1Stot[end]

        # Average [PG1S]tot:
        PG1Save = NumericalIntegration.integrate(r_sol, PG1Stot.*r_sol.^2).*3.0./R^3.0
        PG1Save_out = PG1Save[1]

        return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1Save_out]
    end

    function fbatch_concs(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        p_in = [p_batch[:,i] for i in axes(p_batch,2)]
        pmap_out = @showprogress pmap(pmap_fun_concs, p_in)
        out = hcat(pmap_out...)
        return out
    end

    function fbatch_concs_mt(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
        # @show size(p_batch)
        p_batch = exp.(p_batch)
        # p_batch = 10.0 .^ (p_batch)
        out = zeros(numout, size(p_batch,2))
        prog = Progress(size(out,2))
        Threads.@threads for i in axes(out,2)
            out[:,i] = try
                pmap_fun_concs(p_batch[:,i]; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
            catch
                zeros(size(out[:,i]))
            end
            # @show out[:,i]
            next!(prog)
        end
        return out
    end
end