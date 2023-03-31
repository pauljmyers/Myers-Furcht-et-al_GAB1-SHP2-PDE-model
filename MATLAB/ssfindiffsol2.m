function F = ssfindiffsol2(C,Co,CpEtot,D,k,nargin,Rargin)
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
    kS2f = k(1); %Rate constant for SHP2 binding to pGAB1, um^3/(molec*min)
    kS2r = k(2); %Rate constant for SHP2 unbinding from pGAB1, 1/min
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

    %Define initial concentrations in molecules/um^3 or molecules/um^2 (for
    %surface species)
    CoSFK = Co(1); %Total concentration of SFK
    CoG2 = Co(2); %Total concentration of GRB2
    CoG1 = Co(3); %Total concentration of GAB1
    CoS2 = Co(4); %Total concentration of SHP2
    CoEGFR = Co(5); %Total concentration of EGFR

    m = sqrt(kSi./D_S);

    R = Rargin; %cell radius

    numeqns = nargin; %number of interior spatial nodes to solve equations --> MUST HAVE A VALUE OF AT LEAST 2
    h = R/numeqns; %spatial step size
    r = linspace(0,R,numeqns+1)'; %distance
    
    %Analytical SFK sol'n values
    alpha = kSa*CoSFK*CpEtot/D_S/(cosh(m*R)/R+sinh(m*R)*(-1/(m*R^2)+kSa*CpEtot/(m*D_S*R)));
    CaSFK(r~=0) = alpha*sinh(m.*r(r~=0))./(m.*r(r~=0)); %analytical solution for aSFK
    CaSFK(r==0) = alpha;
    
    %Define equations for GAB1, denoted by C(i,1) --------------
    F(1) = C(2,1)-C(1,1);
    for i = 2:numeqns
        F(i) = D_G1*(1/(r(i)*h)*(C(i+1,1)-C(i-1,1))+(C(i+1,1)-2*C(i,1)+C(i-1,1))./(h^2)) + kG1dp*(CoG1-C(i,1)-CoS2+C(i,2)) - kG1p*C(i,1)*CaSFK(i);
    end
    F(numeqns+1) = C(i+1,1)-C(i,1);
    
    
    %Define equations for SHP2, denoted by C(i,2) --------------
    F(numeqns+2) = C(2,2)-C(1,2);
    for i = 2:numeqns
        F(i+numeqns+1) = D_S2*(1/(r(i)*h)*(C(i+1,2)-C(i-1,2))+(C(i+1,2)-2*C(i,2)+C(i-1,2))./(h^2)) + kS2r*(CoS2-C(i,2)) - kS2f*C(i,2)*(CoG1-C(i,1)-CoS2+C(i,2));
    end
    F(2*numeqns+2) = C(i+1,2)-C(i,2);
    
    GAB1 = C(:,1);
    SHP2 = C(:,2);
    pGAB1 = CoG1 - GAB1 - CoS2 + SHP2;
    G1S2 = CoS2 - SHP2;
    
    
    %Define GAB1 and SHP2 mass balances
    GAB1tot = C(:,1) + pGAB1 + G1S2;
    SHP2tot = C(:,2) + G1S2;
    
    F(2*numeqns+3) = (3/R^3.*trapz(r,GAB1tot.*r.^2) - CoG1).^2; %Difference between average calculated value of total GAB1 and total value of GAB1
    F(2*numeqns+4) = (3/R^3.*trapz(r,SHP2tot.*r.^2) - CoS2).^2; %Difference between average calculated value of total SHP2 and total value of SHP2
end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    