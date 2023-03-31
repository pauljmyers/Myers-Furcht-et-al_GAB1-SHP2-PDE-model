"""
pulsechase_solver(Co::AbstractVector, D::AbstractVector, k::AbstractVector{T};
    R::Float64=10.0, dr::Float64=0.1,
    t_prechase::Float64=5.0, 
    t_chase::Float64=2.0,
    tf::Float64=t_prechase + t_chase,
    Nts::Int64=100, dt::Float64=1.0/(2.0*(maximum(D)/dr.^2 + sum(k)/4))*0.99,
    maxiters=20,
    tol::Float64 = 1.0e-6) where T\n
EGF-gefitinib pulse-chase solver function for the GAB1-SHP2 reaction-diffusion model.\n
...
# Positional arguments
Co = vector of initial species concentrations in the following order: SFKs, GRB2, GAB1, SHP2, and EGFR.
D = vector of diffusivities for cytosolic species in the following order: iSFK, aSFK, GRB2, GRB2-GAB1, GRB2-GAB1-SHP2, GAB1, GAB1-SHP2, SHP2
k = vector of model parameters and kinetic rate constants for model reactions in the following
order: kS2f, kS2r, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp, kSa, kSi, kp, kdp,
kEGFf, kEGFr, [EGF], kdf, kdr.\n
# Keyword arguments
R = simulation radius (radius of the cell)\n
dr = spatial step size\n
t_prechase = EGF treatment/pre-chase time (minutes)\n
t_chase = gefitinib chase time\n
tf = final simulation time\n
dt = time step size\n
dt_save = tf/Nts, time step for saving outputs\n
maxiters = the number of semi-implicit Euler iterations to compute before moving on\n
tol = error tolerance for semi-implicit finite difference scheme at cell membrane.
"""
function pulsechase_solver(Co::AbstractVector, D::AbstractVector, k::AbstractVector{T};
    R::Float64=10.0, dr::Float64=0.1,
    t_prechase::Float64=5.0, 
    t_chase::Float64=2.0,
    tf::Float64=t_prechase + t_chase,
    Nts::Int=100, 
    dt::Float64=1.0/(2.0*(maximum(D)/dr.^2 + sum(k)/4))*0.99,
    dt_save=tf/Nts,
    maxiters=20,
    tol::Float64 = 1.0e-6) where T

    ## Convert input types:
    Co = T[Co...]
    D = T[D...]
    R = T[R][1]
    dr = T[dr][1]
    dt = T[dt][1]

    ## DIFFUSIVITIES:
    D_S = D[1]     # SFKs
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
    # Define initial concentrations in molecules/μm³ (cytoplasmic species)
    # or molecules/μm² (membrane-bound species):
    CoSFK, CoG2, CoG1, CoS2, CoEGFR = Co

    ## Initialize outputs:
    # -- Cytosol:
    iSFK_out = zeros(T, length(r), Nts+1)
    aSFK_out = zeros(T, length(r), Nts+1)
    G2G1_out = zeros(T, length(r), Nts+1)
    PG1S_out = zeros(T, length(r), Nts+1)
    G2PG1_out = zeros(T, length(r), Nts+1)
    G2PG1S_out = zeros(T, length(r), Nts+1)
    PG1_out = zeros(T, length(r), Nts+1)
    GRB2_out = zeros(T, length(r), Nts+1)
    SHP2_out = zeros(T, length(r), Nts+1)
    GAB1_out = zeros(T, length(r), Nts+1)

    iSFK_out[:,1] .= CoSFK
    GRB2_out[:,1] .= CoG2
    SHP2_out[:,1] .= CoS2
    GAB1_out[:,1] .= CoG1
    
    
    # -- Membrane:
    mE_out = zeros(T, Nts+1)
    mES_out = zeros(T, Nts+1)
    mESmES_out = zeros(T, Nts+1)
    E_out = zeros(T, Nts+1)
    pE_out = zeros(T, Nts+1)
    EG2_out = zeros(T, Nts+1)
    EG2G1_out = zeros(T, Nts+1)
    EG2PG1_out = zeros(T, Nts+1)
    EG2PG1S_out = zeros(T, Nts+1)
    EGFR_SHP2_out = zeros(T, Nts+1)

    mE_out[1] = CoEGFR
        
    ## Initialize solution matrices and vectors:
    # -- Cytosol:
    iSFK = zeros(T, length(r), 2)
    aSFK = zeros(T, length(r), 2)
    GAB1 = zeros(T, length(r), 2)
    pGAB1 = zeros(T, length(r), 2)
    GRB2 = zeros(T, length(r), 2)
    G2G1 = zeros(T, length(r), 2)
    G2PG1 = zeros(T, length(r), 2)
    SHP2 = zeros(T, length(r), 2)
    PG1S = zeros(T, length(r), 2)
    G2PG1S = zeros(T, length(r), 2)
    # -- Membrane:
    mE = zeros(T, 1, 2)
    mES = zeros(T, 1, 2)
    mESmES = zeros(T, 1, 2)
    E = zeros(T, 1, 2)
    EG2 = zeros(T, 1, 2)
    EG2G1 = zeros(T, 1, 2)
    EG2PG1 = zeros(T, 1, 2)
    EG2PG1S = zeros(T, 1, 2)

    ## SOLVER
    # Set initial conditions of model species:
    iSFK[:,1] .= CoSFK 
    GAB1[:,1] .= CoG1
    GRB2[:,1] .= CoG2
    SHP2[:,1] .= CoS2
    mE[1] = CoEGFR

    t = T[0.0][1] # Track total elapsed time
    nts = 1     # tracker for number of saved time points
    t_save = dt_save
    # t_out = T[0.0] # Initialize output time vector
    t_out = zeros(T, Nts+1) # Initialize output time vector

    for i in 2:Nt+1 # loop over time points
        if t_prechase+dt > t ≥ t_prechase
            kp = 0.0    # turn off EGFR phosphorylation
        end
        for j in 2:Nr # loop over spatial points
            iSFK[j,2] = (D_S*(1/(r[j]*dr)*(iSFK[j+1,1]-iSFK[j-1,1])+(iSFK[j+1,1]-2.0*iSFK[j,1]+iSFK[j-1,1])/dr^2) + 
                kSi*aSFK[j,1])*dt + iSFK[j,1]

            aSFK[j,2] = (D_S*(1/(r[j]*dr)*(aSFK[j+1,1]-aSFK[j-1,1])+(aSFK[j+1,1]-2.0*aSFK[j,1]+aSFK[j-1,1])/dr^2) - 
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

        # Enforce reactive flux at r = R:
        for i in 1:maxiters # Stay in semi-implicit loop until error tolerance is met
            # Store initial guesses for cytosolic species at membrane and membrane species at new time point:
            cyto_old = [iSFK[Nr+1,2], aSFK[Nr+1,2],
                GAB1[Nr+1,2], pGAB1[Nr+1,2], 
                GRB2[Nr+1,2], G2G1[Nr+1,2], G2PG1[Nr+1,2], 
                SHP2[Nr+1,2], PG1S[Nr+1,2], G2PG1S[Nr+1,2]]
            memb_old = [mE[2], mES[2], mESmES[2], E[2], EG2[2], EG2G1[2], EG2PG1[2], EG2PG1S[2]]
            
            Etot= 2.0*(E[2]+EG2[2]+EG2G1[2]+EG2PG1[2]+EG2PG1S[2])
            iSFK[Nr+1,2] = iSFK[Nr,2]/(1+kSa*Etot*dr/D_S)
            aSFK[Nr+1,2] = aSFK[Nr,2]+kSa*iSFK[Nr+1,2]*Etot*dr/D_S
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
            if error <= tol 
                break
            end
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
        Etot= 2*(E[2]+EG2[2]+EG2G1[2]+EG2PG1[2]+EG2PG1S[2])

        t += dt # increment elapsed time

        # Record concentrations after Nt/Nts time steps:
        if t >= t_save
            nts += 1    # increment number of saved timesteps
            
            iSFK_out[:,nts] = iSFK[:,2]
            aSFK_out[:,nts] = aSFK[:,2]
            GRB2_out[:,nts] = GRB2[:,2]
            GAB1_out[:,nts] = GAB1[:,2]
            SHP2_out[:,nts] = SHP2[:,2]
            G2G1_out[:,nts] = G2G1[:,2]
            G2PG1_out[:,nts] = G2PG1[:,2]
            G2PG1S_out[:,nts] = G2PG1S[:,2]
            PG1_out[:,nts] = pGAB1[:,2]
            PG1S_out[:,nts] = PG1S[:,2]

            EGFR_SHP2_out[nts] = EG2PG1S[2]*100.0/CoEGFR
            mE_out[nts] =  mE[2]
            mES_out[nts] =  mES[2]
            mESmES_out[nts] =  mESmES[2]
            E_out[nts] =  E[2]
            pE_out[nts] =  Etot.*100.0./CoEGFR
            EG2_out[nts] =  EG2[2]
            EG2G1_out[nts] =  EG2G1[2]
            EG2PG1_out[nts] =  EG2PG1[2]
            EG2PG1S_out[nts] =  EG2PG1S[2]

            t_out[nts] = t # Record current time step
            t_save += dt_save
        end
    end

    # -- Calculate total pGAB1 and pGAB1-SHP2 outputs:
    PG1Stot_out = PG1S_out .+ G2PG1S_out
    PG1tot_out = G2PG1_out .+ PG1_out .+ PG1Stot_out

    ## STORE SOLUTIONS FOR OUTPUT:
    sol = (iSFK = iSFK_out, aSFK = aSFK_out, GRB2 = GRB2_out,
        GAB1 = GAB1_out, SHP2 = SHP2_out, G2G1 = G2G1_out,
        G2PG1 = G2PG1_out, G2PG1S = G2PG1S_out, PG1 = PG1_out, 
        PG1S = PG1S_out, PG1tot = PG1tot_out, PG1Stot = PG1Stot_out, 
        EGFR_SHP2 = EGFR_SHP2_out, pE = pE_out, mE = mE_out, 
        mES = mES_out, mESmES = mESmES_out,
        E = E_out, EG2 = EG2_out, EG2G1 = EG2G1_out,
        EG2PG1 = EG2PG1_out, EG2PG1S = EG2PG1S_out)
    return sol, r, t_out, t_prechase, t_chase, dt_save
end