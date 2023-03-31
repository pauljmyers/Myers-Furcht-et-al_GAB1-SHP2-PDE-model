%% Initialize parameters
clear
close all
clc
julia_cols = {'#009AFA','#E26E47','#3FA54E','#c270D2','#AD8F18','#01ABAE','#ED5F92','#C68324','#01A98C','#8F961E','#01A9CD','#9B7EE8','#618CF7','#F16072','#DC65B7','#6D9E33'};


%Diffusivities (µm^2/min)
D_S = 84.0; %SFKs
D_Sa = D_S;  %active SFKs --> set close to 0 for simulating SFKs active only at the membrane (e.g. multiply by 1e-12).
D_G2 = 136.0; %GRB2
D_G2G1 = 62.0; %GRB2-GAB1
D_G2G1S2 = 56.0; %GRB2-GAB1-SHP2
D_G1 = 67.0;  %GAB1
D_G1S2 = 57.0; %GAB1-SHP2
D_S2 = 80.0;  %SHP2

Diffs = [D_S;D_Sa;D_G2;D_G2G1;D_G2G1S2;D_G1;D_G1S2;D_S2]; %diffusivities vector for determining stability criterion

%Kinetic rate constants
kSf = 1.594; %Rate constant for SHP2 binding to pGAB1, µm^3/(molec*min)
kSr = 480.0; %Rate constant for SHP2 unbinding from pGAB1, 1/min
kG1f = 8.842e-4; %Rate constant for GAB1 binding to GRB2, µm^3/(molec*min)
kG1r = 0.1227; %Rate constant GAB1 unbinding from GRB2, 1/min
kG2f =  1.594; %Rate constant for GRB2 binding to EGFR, µm^3/(molec*min)
kG2r = 480.0; %Rate constant for GRB2 unbinding from EGFR, 1/min
kG1p = 1.267;  %GAB1 phosphorylation rate constant, µm^3/(molec*min)
kG1dp = 3.118; %GAB1 dephosphorylation rate constant, min-1
kSa = 0.7924; %SRC activation rate constant µm^3/(molec*min)
kSi = 4.666; %SRC inactivation rate constant, 1/min
kp = 13.84; %EGFR phosphorylation rate constant, µm^2/(molec*min)
kdp = 41.21; %EGFR dephosphorylation rate constant, 1/min
kEGFf = 55.84; %EGF binding rate constant, 1/(µm*min)
kEGFr = 0.1301; %EGF unbinding rate constant, 1/min
EGF = 0.00167; %EGF concentration, µm
kdf = 1.2; %Dimer on rate constant, µm^2/(molec*min)
kdr = 0.456; %Dimer off rate constant, 1/min


%Kinetic rate constant vector for determining stability criterion and
%feeding to solver function:
kvals = [kSf;kSr;kG1f;kG1r;kG2f;kG2r;kG1p;kG1dp;kSa;kSi;kp;kdp;kEGFf;kEGFr;EGF;kdf;kdr];

%%Define solver constants:
dr = 0.1; %length of each depth step
R = 10; % Radius of cell
Nr = R/dr; % Choose the number of depth steps
Nts = 100; % Number of time points to store

%Set time step based on stability criterion
maxdt = 1/(2.*(max(Diffs)/dr.^2+sum(kvals)/4)) %Stability criterion for max dt --> based on criterion from Bieniasz, J. Elctroanal. Chem., 1992
dt = 0.99*maxdt; %Length of each time step in seconds
tf = 5.0; %final integration time

volCF = 1/(4/3*pi*R^3);  %conversion factor from cell^-1 to umol/µm^3 for a spherical cell
surfCF = 1/(4*pi*R^2); %conversion factor from cell^-1 to umol/µm^2 for a spherical cell

%Define initial concentrations in molecules/µm^3 or molecules/µm^2 (for
%surface species)
CoSFK = 145; %Total concentration of SFK (molec/um3)
CoG2 = 145; %Total concentration of GRB2 (molec/um3)
CoG1 = 145; %Total concentration of GAB1 (molec/um3)
CoS2 = 145; %Total concentration of SHP2 (molec/um3)
CoEGFR = 6e5*surfCF; %Total concentration of EGFR (molec/um2)
Co = [CoSFK; CoG2; CoG1; CoS2; CoEGFR]; % Vector of initial concentrations
% CpEtot = 0.63028*CoEGFR; %Steady-state concentration of pEGFR

%% Solve FULL PDE SYSTEM
tic
[sol,r_sol,t] = basepdesolver(Co,Diffs,kvals,R,dr,tf,Nts);
toc
Nt = length(t);

%%
%Steady-state concentration of pEGFR:
CpEtot = sol.pE(end)/100*CoEGFR;

%Reduce number of nodes for plotting:
ns = 20; %number of plotting nodes
aSFK_plot2 = sol.aSFK(1:Nr/ns:end,1:Nr/ns:end); %aSFKs
iSFK_plot2 = CoSFK-aSFK_plot2;
GAB1_plot2 = sol.GAB1(1:Nr/ns:end,1:Nr/ns:end); % GAB1
pGAB1_plot2 = sol.PG1tot(1:Nr/ns:end,1:Nr/ns:end); %total pGAB1
pGAB1_SHP2_plot2 = sol.PG1Stot(1:Nr/ns:end,1:Nr/ns:end); %total GAB1-SHP2
SHP2_plot2 = sol.SHP2(1:Nr/ns:end,1:Nr/ns:end); % SHP2
t2=t(1:Nr/ns:end); %new time vector for plotting
x2=r_sol(1:Nr/ns:end); %new distance vector for plotting


%Collect full PDE solutions
plotC = [sol.aSFK(:,end),...
    sol.iSFK(:,end),...
    sol.GAB1(:,end)+ sol.G2G1(:,end),...
    sol.PG1tot(:,end),...
    sol.PG1Stot(:,end),...
    sol.SHP2(:,end)]; %./volCF;

%% Analytical solutions for SFKs
r = linspace(0,R)'; %Solution domain for analytical SFK solutions
rspan = [0,R]; %Solution domain for BVP routine

m = sqrt(kSi./D_S);
CaS = kSa*CoSFK*CpEtot/D_S/(cosh(m*R)/R+sinh(m*R)*(-1/(m*R^2)+kSa*CpEtot/(m*D_S*R)))*sinh(m.*r)./(m.*r); %analytical solution for aSFK
CiS = CoSFK - CaS; %analytical solution for iSFK

%% Solve 4-ODE steady-state system using finite differences method with fsolve (no time point evaluations, i.e. direct SS calculation)
clc
R4 = 10.0; %cell radius
n4 = 100; %number of interior spatial nodes to obtain solution

findifffunc4 = @(Cguess) ssfindiffsol4(Cguess,Co,CpEtot,Diffs,kvals,n4,R4);
Cguess = 75*ones(n4+1,4); %initial guess matrix
options = optimoptions('fsolve','Display','off','Algorithm','trust-region'); %,'SubproblemAlgorithm','cg');

% Csol4 = fsolve(findifffunc4, Cguess,options)./volCF;
Csol4 = fsolve(findifffunc4, Cguess,options);

PG1sol4 = Csol4(:,2)+Csol4(:,3);

%% Solve 2-ODE steady-state system using finite differences method with fsolve (no time point evaluations, i.e. direct SS calculation)
%Solve 2-ODE system:
findifffunc2 = @(Cguess) ssfindiffsol2(Cguess,Co,CpEtot,Diffs,kvals,n4,R4);
Cguess = 75*ones(n4+1,2); %initial guess matrix
% Csol2 = fsolve(findifffunc2, Cguess,options)./volCF;
Csol2 = fsolve(findifffunc2, Cguess,options);

G1sol2 = Csol2(:,1);
S2sol2 = Csol2(:,2);
% PG1Ssol2 = CoS2./volCF - S2sol2;
% PG1sol2tot = CoG1./volCF - G1sol2 - CoS2./volCF + S2sol2 + PG1Ssol2;
PG1Ssol2 = CoS2 - S2sol2;
PG1sol2tot = CoG1 - G1sol2 - CoS2 + S2sol2 + PG1Ssol2;

%% Solve 1-ODE steady-state system using finite differences method with fsolve (no time point evaluations, i.e. direct SS calculation)
% Solve 1-ODE system:
findifffunc1 = @(Cguess) ssfindiffsol1(Cguess,Co,CpEtot,Diffs,kvals,n4,R4);
Cguess = 75*ones(n4+1,1); %initial guess matrix
% Csol1 = fsolve(findifffunc1, Cguess,options)./volCF;
Csol1 = fsolve(findifffunc1, Cguess,options);

G1sol1 = Csol1;
% S2sol1 = (-(CoG1./volCF-G1sol1-CoS2./volCF+kSr/kSf./volCF)+sqrt((CoG1./volCF-G1sol1-CoS2./volCF+kSr/kSf./volCF).^2+4*kSr/kSf./volCF*CoS2./volCF))./2;
% PG1Ssol1 = CoS2./volCF - S2sol1;
% PG1sol1tot = CoG1./volCF  - G1sol1;
S2sol1 = (-(CoG1-G1sol1-CoS2+kSr/kSf)+sqrt((CoG1-G1sol1-CoS2+kSr/kSf).^2+4*kSr/kSf*CoS2))./2;
PG1Ssol1 = CoS2 - S2sol1;
PG1sol1tot = CoG1  - G1sol1;

%% Plot solutions
close all
lw = 1.0;
timelabel = num2str(tf);
namesC = {'aSFK';'iSFK';'GAB1';'pGAB1';'GAB1-SHP2';'SHP2'};

%%% Analytical and numerical SFK solutions
step = 9;
ind = [1:step:length(r),length(r)];
figure; colororder(julia_cols)
% plot(r_sol,plotC(:,1),r_sol,plotC(:,2),r(ind),CaS(ind)./volCF,'^',r(ind),CiS(ind)./volCF,'sq','LineWidth',lw)
plot(r_sol,plotC(:,1),r_sol,plotC(:,2),r(ind),CaS(ind),'^',r(ind),CiS(ind),'sq','LineWidth',lw)
xlabel('Distance from cell center (µm)')
ylabel('Concentration (molec•µm^{-3})')
title(['SFKs, ',timelabel,' min EGF for PDE solution'])
ylim([0, max(plotC(:,1:2),[],'all')*1.05])
xlim([0 R])
xticks(0:R/2:R)
% yticks(0:1/4*145:145)
box off
set(gca,'TickDir','out','LineWidth',lw,'FontSize',8)
legend('aSFK_{PDE}','iSFK_{PDE}','aSFK_{analytic}','iSFK_{analytic}','location','best')
ax1=gca;
exportgraphics(ax1,'images/steady-state_comps_SFKs.pdf','ContentType','vector','Resolution',60)


%% GAB1 -------------------------------------------------
%%% PDE/ODE solutions
h4 = R4/n4; %spatial step size
rsol4 = (0:h4:R4)'; %spatial grid

figure; colororder(julia_cols)
plot(r_sol,plotC(:,3),rsol4,Csol4(:,1),'-.',rsol4,G1sol2,':',rsol4,G1sol1,'--','LineWidth',lw)
xlabel('Distance from cell center (µm)')
ylabel('Concentration (molec•µm^{-3})')
title(['GAB1, ',timelabel,' min EGF for PDE solution'])
ylim([min(plotC(:,3))*0.9, max(plotC(:,3))*1.1])
xlim([0 R])
xticks(0:R/2:R)
% yticks(0:1/2*75:75)
box off
set(gca,'TickDir','out','LineWidth',lw,'FontSize',8)
legend('GAB1_{PDE}','GAB1_{4ODE}','GAB1_{2ODE}','GAB1_{1ODE}','location','best')
ax2=gca;
exportgraphics(ax2,'images/steady-state_comps_GAB1.pdf','ContentType','vector')

%% pGAB1 -------------------------------------------------
figure; colororder(julia_cols)
plot(r_sol,plotC(:,4),rsol4,PG1sol4,'-.',rsol4,PG1sol2tot,':',rsol4,PG1sol1tot,'--','LineWidth',lw)
xlabel('Distance from cell center (µm)')
ylabel('Concentration (molec•µm^{-3})')
title(['Total pGAB1, ',timelabel,' min EGF for PDE solution'])
ylim([min(plotC(:,4))*0.95, max(plotC(:,4))*1.05])
xlim([0 R])
xticks(0:R/2:R)
% yticks(0:1/4*145:145)
box off
set(gca,'TickDir','out','LineWidth',lw,'FontSize',8)
legend('pGAB1_{PDE}','pGAB1_{4ODE}','pGAB1_{2ODE}','pGAB1_{1ODE}','location','best')
ax3=gca;
exportgraphics(ax3,'images/steady-state_comps_pGAB1.pdf','ContentType','vector')

%% GAB1-SHP2 -------------------------------------------------
figure; colororder(julia_cols)
plot(r_sol,plotC(:,5),rsol4,Csol4(:,3),'-.',rsol4,PG1Ssol2,':',rsol4,PG1Ssol1,'--','LineWidth',lw)
xlabel('Distance from cell center (µm)')
ylabel('Concentration (molec•µm^{-3})')
title(['GAB1-SHP2, ',timelabel,' min EGF for PDE solution'])
ylim([min(plotC(:,5))*0.95, max(plotC(:,5))*1.05])
xlim([0 R])
xticks(0:R/2:R)
% yticks(0:1/4*110:110)
box off
set(gca,'TickDir','out','LineWidth',lw,'FontSize',8)
legend('GAB1-SHP2_{PDE}','GAB1-SHP2_{4ODE}','GAB1-SHP2_{2ODE}','GAB1-SHP2_{1ODE}','location','best')
ax4=gca;
exportgraphics(ax4,'images/steady-state_comps_GAB1-SHP2.pdf','ContentType','vector')

%% SHP2 -------------------------------------------------
figure; colororder(julia_cols)
plot(r_sol,plotC(:,6),rsol4,Csol4(:,4),'-.',rsol4,S2sol2,':',rsol4,S2sol1,'--','LineWidth',lw)
xlabel('Distance from cell center (µm)')
% ylabel('Concentration (cell^{-1})')
ylabel('Concentration (molec•µm^{-3})')
title(['SHP2, ',timelabel,' min EGF for PDE solution'])
ylim([min(plotC(:,6))*0.99, max(plotC(:,6))*1.01])
xlim([0 R])
xticks(0:R/2:R)
% yticks(0:1/4*110:110)
box off
set(gca,'TickDir','out','LineWidth',lw,'FontSize',8)
legend('SHP2_{PDE}','SHP2_{4ODE}','SHP2_{2ODE}','SHP2_{1ODE}','location','best')
ax5=gca;
exportgraphics(ax5,'images/steady-state_comps_SHP2.pdf','ContentType','vector')








