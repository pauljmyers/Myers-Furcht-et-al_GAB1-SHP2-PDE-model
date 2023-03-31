function [sol,r,t_out] = basepdesolver(Co,D,k,R,dr,tf,Nts,dt,tol,maxiters)
% Solver function for the base model. The inputs are as follows:
% Co = vector of initial species concentrations in the following order:
% SFKs, GRB2, GAB1, SHP2, and EGFR.
% D = vector of diffusivities for cytosolic species in the following order:
% iSFK, aSFK, GRB2, GRB2-GAB1, GRB2-GAB1-SHP2, GAB1, GAB1-SHP2, SHP2
% k = vector of model parameters and kinetic rate constants for model reactions in the following
% order: kSf, kSr, kG1f, kG1r, kG2f, kG2r, kG1p, kG1dp, kSa, kSi, kp, kdp,
% kEGFf, kEGFr, [EGF], kdf, kdr
% R = simulation radius (radius of the cell)
% dr = spatial step size
% tf = final simulation time
% Nts = number of time points to store for the solution, evenly spaced
% across the simulation time
% dt = time step size
% tol = error tolerance for semi-implicit finite difference scheme at cell membrane.
% "tol" is an optional parameter with a default value of 1.0e-6.


if (~exist('R', 'var'))
    R = 10.0;
end


if (~exist('dr', 'var'))
    dr = 0.1;
end


if (~exist('tf', 'var'))
    tf = 5.0;
end


if (~exist('Nts', 'var'))
    Nts = 100;
end

if (~exist('dt', 'var'))
    dt = 1/(2.*(max(D)/dr.^2+sum(k)/4));
end

if (~exist('tol', 'var'))
    tol = 1e-6;
end

if (~exist('maxiters', 'var'))
    maxiters = 100;
end

%DIFFUSIVITIES:
D_S = D(1); %SFKs
D_Sa = D(2);  %active SFKs --> set close to 0 for simulating SFKs active only at the membrane (e.g. multiply by 1e-12).
D_G2 = D(3); %GRB2
D_G2G1 = D(4); %GRB2-GAB1
D_G2G1S2 = D(5); %GRB2-GAB1-SHP2
D_G1 = D(6);  %GAB1
D_G1S2 = D(7); %GAB1-SHP2
D_S2 = D(8);  %SHP2

%KINETIC RATE CONSTANTS:
kSf = k(1); %Rate constant for SHP2 binding to pGAB1, um^3/(molec*min)
kSr = k(2); %Rate constant for SHP2 unbinding from pGAB1, 1/min
kG1f = k(3); %Rate constant for GAB1 binding to GRB2, um^3/(molec*min)
kG1r = k(4); %Rate constant GAB1 unbinding from GRB2, 1/min
kG2f = k(5); %Rate constant for GRB2 binding to EGFR, um^3/(molec*min)
kG2r = k(6); %Rate constant for GRB2 unbinding from EGFR, 1/min
kG1p = k(7);  %GAB1 phosphorylation rate constant, um^3/(molec*min)
kG1dp = k(8); %GAB1 dephosphorylation rate constant, min-1
kSa = k(9); %SRC activation rate constant um^3/(molec*min)
kSi = k(10); %SRC inactivation rate constant, 1/min
kp = k(11); %EGFR phosphorylation rate constant, um^2/(molec*min)
kdp = k(12); %EGFR dephosphorylation rate constant, 1/min
kEGFf = k(13); %EGF binding rate constant, 1/(uM*min)
kEGFr = k(14); %EGF unbinding rate constant, 1/min
EGF = k(15); %EGF concentration, uM
kdf = k(16); %Dimer on rate constant, um^2/(molec*min)
kdr = k(17); %Dimer off rate constant, 1/min

%SOLVER OPTIONS:
Nr = ceil(R/dr); % Number of spatial steps
Nt = ceil(tf/dt); % Choose the number of time steps
dt_save = tf/Nts; % Time step for saving outputs
r = [0:dr:R]'; % Distance grid

%%%SOLVER SETUP:
%Define initial concentrations in molecules/um^3 or molecules/um^2 (for
%surface species)
CoSFK = Co(1); %Total concentration of SFK
CoG2 = Co(2); %Total concentration of GRB2
CoG1 = Co(3); %Total concentration of GAB1
CoS2 = Co(4); %Total concentration of SHP2
CoEGFR = Co(5); %Total concentration of EGFR

%Initialize output matrices:
iSFK_out = zeros(length(r), Nts+1);
aSFK_out = zeros(length(r), Nts+1);
G2G1_out = zeros(length(r), Nts+1);
GRB2_out = zeros(length(r), Nts+1);
PG1S_out = zeros(length(r), Nts+1);
SHP2_out = zeros(length(r), Nts+1);
SHP2_out(:,1) = CoS2;
GAB1_out = zeros(length(r), Nts+1);
GAB1_out(:,1) = CoG1;
G2PG1_out = zeros(length(r), Nts+1);
G2PG1S_out = zeros(length(r), Nts+1);
PG1_out = zeros(length(r), Nts+1);
PG1tot_out = zeros(length(r), Nts+1);
PG1Stot_out = zeros(length(r), Nts+1);

EGFR_SHP2_out = zeros(1, Nts+1);
pE_out = zeros(1, Nts+1);
mE_out = zeros(1, Nts+1);
mES_out = zeros(1, Nts+1);
mESmES_out = zeros(1, Nts+1); 
E_out = zeros(1, Nts+1); 
EG2_out = zeros(1, Nts+1);
EG2G1_out = zeros(1, Nts+1);
EG2PG1_out = zeros(1, Nts+1);
EG2PG1S_out = zeros(1, Nts+1);
    
%Initialize solution matrices
iSFK = zeros(length(r),2);
aSFK = zeros(length(r),2);
GAB1 = zeros(length(r),2);
pGAB1 = zeros(length(r),2);
GRB2 = zeros(length(r),2);
G2G1 = zeros(length(r),2);
G2PG1 = zeros(length(r),2);
SHP2 = zeros(length(r),2);
PG1S = zeros(length(r),2);
G2PG1S = zeros(length(r),2);

initialE = CoEGFR;
mE = zeros(2);
mES = zeros(2);
mESmES = zeros(2);
E = zeros(2);
EG2 = zeros(2);
EG2G1 = zeros(2);
EG2PG1 = zeros(2);
EG2PG1S = zeros(2);

%%%SOLVER
%Set initial conditions of model species:
iSFK(:,1) = CoSFK; 
aSFK(:,1) = 0;
GAB1(:,1) = CoG1;
pGAB1(:,1)= 0;
GRB2(:,1) = CoG2;
G2G1(:,1) = 0;
G2PG1(:,1)= 0;
SHP2(:,1) = CoS2;
PG1S(:,1) = 0;
G2PG1S(:,1)=0;
mE(1) = CoEGFR;
mES(1) = 0;
mESmES(1) = 0;
E(1) = 0;
E(1) = 0;
EG2(1) = 0;
EG2G1(1) = 0;
EG2PG1(1) = 0;
EG2PG1S(1) = 0;

t = 0.0; % Track total elapsed time
nts = 1;     % tracker for number of saved time points
t_save = dt_save;
t_out = zeros(1, Nts+1); % Initialize output time vector
    
for i=2:Nt+1 %loop over time points
    for j=2:Nr %loop over spatial points
        iSFK(j,2) = (D_S*(1/(r(j)*dr)*(iSFK(j+1,1)-iSFK(j-1,1))+(iSFK(j+1,1)-2.0*iSFK(j,1)+iSFK(j-1,1))/dr^2) + ...
            kSi*aSFK(j,1))*dt + iSFK(j,1);
        aSFK(j,2) = (D_Sa*(1/(r(j)*dr)*(aSFK(j+1,1)-aSFK(j-1,1))+(aSFK(j+1,1)-2.0*aSFK(j,1)+aSFK(j-1,1))/dr^2) - ...
            kSi*aSFK(j,1))*dt + aSFK(j,1);
        GAB1(j,2) = (D_G1*(1/(r(j)*dr)*(GAB1(j+1,1)-GAB1(j-1,1))+(GAB1(j+1,1)-2.0*GAB1(j,1)+GAB1(j-1,1))/dr^2) - ...
            kG1f*GAB1(j,1)*GRB2(j,1) + kG1r*G2G1(j,1) - kG1p*aSFK(j,1)*GAB1(j,1) + kG1dp*pGAB1(j,1))*dt + GAB1(j,1);
        pGAB1(j,2)= (D_G1*(1/(r(j)*dr)*(pGAB1(j+1,1)-pGAB1(j-1,1))+(pGAB1(j+1,1)-2.0*pGAB1(j,1)+pGAB1(j-1,1))/dr^2) - ...
            kG1f*pGAB1(j,1)*GRB2(j,1) + kG1r*G2PG1(j,1) + kG1p*aSFK(j,1)*GAB1(j,1) - kG1dp*pGAB1(j,1) - kSf*SHP2(j,1)*pGAB1(j,1) + kSr*PG1S(j,1))*dt + pGAB1(j,1);
        GRB2(j,2) = (D_G2*(1/(r(j)*dr)*(GRB2(j+1,1)-GRB2(j-1,1))+(GRB2(j+1,1)-2.0*GRB2(j,1)+GRB2(j-1,1))/dr^2) - ...
            kG1f*GAB1(j,1)*GRB2(j,1) + kG1r*G2G1(j,1) - kG1f*pGAB1(j,1)*GRB2(j,1) + kG1r*G2PG1(j,1) - kG1f*GRB2(j,1)*PG1S(j,1) + kG1r*G2PG1S(j,1))*dt + GRB2(j,1);
        G2G1(j,2) = (D_G2G1*(1/(r(j)*dr)*(G2G1(j+1,1)-G2G1(j-1,1))+(G2G1(j+1,1)-2.0*G2G1(j,1)+G2G1(j-1,1))/dr^2) + ...
            kG1f*GAB1(j,1)*GRB2(j,1) - kG1r*G2G1(j,1) - kG1p*aSFK(j,1)*G2G1(j,1) + kG1dp*G2PG1(j,1))*dt + G2G1(j,1);
        G2PG1(j,2)= (D_G2G1*(1/(r(j)*dr)*(G2PG1(j+1,1)-G2PG1(j-1,1))+(G2PG1(j+1,1)-2.0*G2PG1(j,1)+G2PG1(j-1,1))/dr^2) + ...
            kG1f*pGAB1(j,1)*GRB2(j,1) - kG1r*G2PG1(j,1) + kG1p*aSFK(j,1)*G2G1(j,1) - kG1dp*G2PG1(j,1) - kSf*SHP2(j,1)*G2PG1(j,1) + kSr*G2PG1S(j,1))*dt + G2PG1(j,1);
        SHP2(j,2) = (D_S2*(1/(r(j)*dr)*(SHP2(j+1,1)-SHP2(j-1,1))+(SHP2(j+1,1)-2.0*SHP2(j,1)+SHP2(j-1,1))/dr^2) - ...
            kSf*SHP2(j,1)*pGAB1(j,1) + kSr*PG1S(j,1) - kSf*SHP2(j,1)*G2PG1(j,1) + kSr*G2PG1S(j,1))*dt + SHP2(j,1);
        PG1S(j,2) = (D_G1S2*(1/(r(j)*dr)*(PG1S(j+1,1)-PG1S(j-1,1))+(PG1S(j+1,1)-2.0*PG1S(j,1)+PG1S(j-1,1))/dr^2) + ...
            kSf*SHP2(j,1)*pGAB1(j,1) - kSr*PG1S(j,1) - kG1f*GRB2(j,1)*PG1S(j,1) + kG1r*G2PG1S(j,1))*dt + PG1S(j,1);
        G2PG1S(j,2)=(D_G2G1S2*(1/(r(j)*dr)*(G2PG1S(j+1,1)-G2PG1S(j-1,1))+(G2PG1S(j+1,1)-2.0*G2PG1S(j,1)+G2PG1S(j-1,1))/dr^2) + ...
            kG1f*GRB2(j,1)*PG1S(j,1) - kG1r*G2PG1S(j,1) + kSf*SHP2(j,1)*G2PG1(j,1) - kSr*G2PG1S(j,1))*dt + G2PG1S(j,1);
    end

    %Enforce zero flux at r = 0:
    iSFK(1,2) = iSFK(2,2);
    aSFK(1,2) = aSFK(2,2);
    GAB1(1,2) = GAB1(2,2);
    pGAB1(1,2)= pGAB1(2,2);
    GRB2(1,2) = GRB2(2,2);
    G2G1(1,2) = G2G1(2,2);
    G2PG1(1,2)= G2PG1(2,2);
    SHP2(1,2) = SHP2(2,2);
    PG1S(1,2) = PG1S(2,2);
    G2PG1S(1,2) = G2PG1S(2,2);

    %%Enforce reactive flux at r = R:
%     error = tol*2; % Initialize error with a value > tol
%     while error > tol %Stay in semi-implicit loop until error tolerance is met
    for j = 1:maxiters
        % Store initial guesses for cytosolic species at membrane and membrane species at new time point:
        cyto_old = [iSFK(Nr+1,2); aSFK(Nr+1,2); GAB1(Nr+1,2); pGAB1(Nr+1,2); GRB2(Nr+1,2); G2G1(Nr+1,2); G2PG1(Nr+1,2); SHP2(Nr+1,2); PG1S(Nr+1,2); G2PG1S(Nr+1,2)];
        memb_old = [mE(2); mES(2); mESmES(2); E(2); EG2(2); EG2G1(2); EG2PG1(2); EG2PG1S(2)];

        Etot=2*(E(2)+EG2(2)+EG2G1(2)+EG2PG1(2)+EG2PG1S(2));
        iSFK(Nr+1,2) = iSFK(Nr,2)/(1+kSa*Etot*dr/D_S);
        aSFK(Nr+1,2) = aSFK(Nr,2)+kSa*iSFK(Nr+1,2)*Etot*dr/D_Sa;
        GAB1(Nr+1,2) = (kG1r*EG2G1(2)*dr/D_G1+GAB1(Nr,2))/(1+kG1f*EG2(2)*dr/D_G1);
        pGAB1(Nr+1,2)= (kG1r*EG2PG1(2)*dr/D_G1+pGAB1(Nr,2))/(1+kG1f*EG2(2)*dr/D_G1);
        GRB2(Nr+1,2) = (kG2r*EG2(2)*dr/D_G2+GRB2(Nr,2))/(1+kG2f*E(2)*dr/D_G2);
        G2G1(Nr+1,2) = (kG2r*EG2G1(2)*dr/D_G2G1+G2G1(Nr,2))/(1+kG2f*E(2)*dr/D_G2G1);
        G2PG1(Nr+1,2) =(kG2r*EG2PG1(2)*dr/D_G2G1+G2PG1(Nr,2))/(1+kG2f*E(2)*dr/D_G2G1);
        SHP2(Nr+1,2) =  (kSr*EG2PG1S(2)*dr/D_S2+SHP2(Nr,2))/(1+kSf*EG2PG1(2)*dr/D_S2);
        PG1S(Nr+1,2) = (kG1r*EG2PG1S(2)*dr/D_G1S2+PG1S(Nr,2))/(1+kG1f*EG2(2)*dr/D_G1S2);
        G2PG1S(Nr+1,2)=(kG2r*EG2PG1S(2)*dr/D_G2G1S2+G2PG1S(Nr,2))/(1+kG2f*E(2)*dr/D_G2G1S2);


        %Semi-implicit Euler
        %Solve for concentrations of surface species:
        mE(2)= (-kEGFf*EGF*mE(1)+kEGFr*mES(1))*dt + mE(1);
        mES(2) = (kEGFf*EGF*mE(1)-kEGFr*mES(1)-2*kdf*mES(1)*mES(1)+2*kdr*mESmES(1))*dt + mES(1);
        mESmES(2) =(kdf*mES(1)*mES(1)-kdr*mESmES(1)-kp*mESmES(1)+kdp*E(1))*dt + mESmES(1);
        E(2) = (kp*mESmES(1)-kdp*E(1)-kG2f*E(1)*GRB2(Nr+1,2)+kG2r*EG2(1)-kG2f*E(1)*G2G1(Nr+1,2)+kG2r*EG2G1(1)-kG2f*E(1)*G2PG1(Nr+1,2)+kG2r*EG2PG1(1)-kG2f*E(1)*G2PG1S(Nr+1,2)+kG2r*EG2PG1S(1))*dt + E(1);
        EG2(2) = (kG2f*GRB2(Nr+1,2)*E(1)-kG2r*EG2(1)-kG1f*GAB1(Nr+1,2)*EG2(1)+kG1r*EG2G1(1)-kG1f*pGAB1(Nr+1,2)*EG2(1)+kG1r*EG2PG1(1)-kG1f*PG1S(Nr+1,2)*EG2(1)+kG1r*EG2PG1S(1))*dt + EG2(1);
        EG2G1(2) = (kG2f*G2G1(Nr+1,2)*E(1)-kG2r*EG2G1(1)+kG1f*GAB1(Nr+1,2)*EG2(1)-kG1r*EG2G1(1))*dt + EG2G1(1);
        EG2PG1(2) = (kG2f*G2PG1(Nr+1,2)*E(1)-kG2r*EG2PG1(1)+kG1f*pGAB1(Nr+1,2)*EG2(1)-kG1r*EG2PG1(1)-kSf*SHP2(Nr+1,2)*EG2PG1(1)+kSr*EG2PG1S(1))*dt + EG2PG1(1);
        EG2PG1S(2) = (kSf*SHP2(Nr+1,2)*EG2PG1(1)-kSr*EG2PG1S(1)+kG1f*PG1S(Nr+1,2)*EG2(1)-kG1r*EG2PG1S(1)+kG2f*G2PG1S(Nr+1,2)*E(1)-kG2r*EG2PG1S(1))*dt + EG2PG1S(1);

        % Store new estimates for cytosolic species at membrane and membrane species at new time point:
        cyto_new = [iSFK(Nr+1,2); aSFK(Nr+1,2); GAB1(Nr+1,2); pGAB1(Nr+1,2); GRB2(Nr+1,2); G2G1(Nr+1,2); G2PG1(Nr+1,2); SHP2(Nr+1,2); PG1S(Nr+1,2); G2PG1S(Nr+1,2)];
        memb_new = [mE(2); mES(2); mESmES(2); E(2); EG2(2); EG2G1(2); EG2PG1(2); EG2PG1S(2)];

        % Calculate error between new and old estimates for membrane concentrations at new time point:
        error = max([1 - cyto_new./cyto_old; 1 - memb_new./memb_old]);
        if error <= tol
            break
        else
        end
    end            

    %Update solutions for next iteration:
    iSFK(:,1) = iSFK(:,2);
    aSFK(:,1) = aSFK(:,2);
    GAB1(:,1) = GAB1(:,2);
    pGAB1(:,1) = pGAB1(:,2);
    GRB2(:,1) = GRB2(:,2);
    G2G1(:,1) = G2G1(:,2);
    G2PG1(:,1) = G2PG1(:,2);
    SHP2(:,1) = SHP2(:,2);
    PG1S(:,1) = PG1S(:,2);
    G2PG1S(:,1) = G2PG1S(:,2);

    mE(1) = mE(2);
    mES(1) = mES(2);
    mESmES(1) = mESmES(2);
    E(1) = E(2);
    EG2(1) = EG2(2);
    EG2G1(1) = EG2G1(2);
    EG2PG1(1) = EG2PG1(2);
    EG2PG1S(1) = EG2PG1S(2);


    t = t + dt;

     %Record concentrations after Nt/Nts time steps
     if t >= t_save
         nts = nts + 1;
         iSFK_out(:,nts) = iSFK(:,2);
         aSFK_out(:,nts) = aSFK(:,2);
         GRB2_out(:,nts) = GRB2(:,2);
         GAB1_out(:,nts) = GAB1(:,2);
         SHP2_out(:,nts) = SHP2(:,2);
         G2G1_out(:,nts) = G2G1(:,2);
         G2PG1_out(:,nts) = G2PG1(:,2);
         G2PG1S_out(:,nts) = G2PG1S(:,2);
         PG1_out(:,nts) = pGAB1(:,2);
         PG1S_out(:,nts) = PG1S(:,2);
         PG1tot_out(:,nts) = (G2PG1(:,2)+pGAB1(:,2)+PG1S(:,2)+G2PG1S(:,2));
         PG1Stot_out(:,nts) = (PG1S(:,2)+G2PG1S(:,2));
         
         EGFR_SHP2_out(nts) = EG2PG1S(2)*100/initialE;
         pE_out(nts) = Etot*100/initialE;
         mE_out(nts) = mE(2);
         mES_out(nts) = mES(2);
         mESmES_out(nts) = mESmES(2);
         E_out(nts) = E(2);
         EG2_out(nts) = EG2(2);
         EG2G1_out(nts) = EG2G1(2);
         EG2PG1_out(nts) = EG2PG1(2);
         EG2PG1S_out(nts) = EG2PG1S(2);

         t_out(nts) = t;

         t_save = t_save + dt_save;
     end

end

%%STORE SOLUTIONS FOR OUTPUT:
sol.iSFK = iSFK_out;
sol.aSFK = aSFK_out;
sol.GRB2 = GRB2_out;
sol.GAB1 = GAB1_out;
sol.SHP2 = SHP2_out;
sol.G2G1 = G2G1_out;
sol.G2PG1 = G2PG1_out;
sol.G2PG1S = G2PG1S_out;
sol.PG1 = PG1_out;
sol.PG1S = PG1S_out;
sol.PG1tot = PG1tot_out;
sol.PG1Stot = PG1Stot_out;
sol.EGFR_SHP2 = EGFR_SHP2_out;
sol.pE = pE_out;
sol.mE = mE_out;
sol.mES = mES_out;
sol.mESmES = mESmES_out;
sol.E = E_out;
sol.EG2 = EG2_out;
sol.EG2G1 = EG2G1_out;
sol.EG2PG1 = EG2PG1_out;
sol.EG2PG1S = EG2PG1S_out;
end