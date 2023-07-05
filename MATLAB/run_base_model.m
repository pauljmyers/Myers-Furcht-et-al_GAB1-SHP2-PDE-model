%%% Initialize parameters
clear
close all
clc
julia_cols = {'#009AFA','#E26E47','#3FA54E','#c270D2','#AD8F18','#01ABAE','#ED5F92','#C68324','#01A98C','#8F961E','#01A9CD','#9B7EE8','#618CF7','#F16072','#DC65B7','#6D9E33'};

%% Base model parameters
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

%% Load parameter ensemble
T = readtable('parameter_ensemble.csv');
Dmat = table2array(T(:,1:7));   % diffusivities
Dmat = [T.Dsfk Dmat];   % add column for iSFK
kmat = table2array(T(:,8:end)); % kinetic parameters


%% Define other model constants and parameters:
dr = 0.2; %length of each depth step
R = 10; % Radius of cell
Nr = R/dr; % Choose the number of depth steps
Nts = 100; % Number of time points to store

%Set time step based on stability criterion
maxdt = 1/(2.*(max(Diffs)/dr.^2+sum(kvals)/4)); %Stability criterion for max dt --> based on criterion from Bieniasz, J. Elctroanal. Chem., 1992
dt = 0.99*maxdt; %Length of each time step in seconds
tf = 5.0; %final integration time

volCF = 1/(4/3*pi*R^3);  %conversion factor from cell^-1 to µmol/µm^3 for a spherical cell
surfCF = 1/(4*pi*R^2); %conversion factor from cell^-1 to µmol/µm^2 for a spherical cell

%Define initial concentrations in molecules/µm^3 or molecules/µm^2 (for
%surface species)
CoSFK = 6e5*volCF; %Total concentration of SFK (molec/µm^3)
CoG2 = 6e5*volCF; %Total concentration of GRB2 (molec/µm^3)
CoG1 = 6e5*volCF; %Total concentration of GAB1 (molec/µm^3)
CoS2 = 6e5*volCF; %Total concentration of SHP2 (molec/µm^3)
CoEGFR = 6e5*surfCF; %Total concentration of EGFR (molec/µm^2)
Co = [CoSFK; CoG2; CoG1; CoS2; CoEGFR]; % Vector of initial concentrations

%% Test PDE solver
% Spherical coordinates:
tic
[sol,r,t] = basepdesolver(Co,Diffs,kvals,R,dr,tf,Nts);
toc

% Rectangular coordinates:
tic
[sol_rect,r_rect,t_rect] = basepdesolver_rect(Co,Diffs,kvals,R,dr,tf,Nts);
toc

%% Run parameter ensemble simulations:
% These model simulations use the parameter distributions generated from
% the Julia code associated with this repository/project (which are loaded
% from the CSV file "parameter_ensemble.csv".

N_spls = 2000; % number of samples to draw from parameter distributions for ensemble simulations
idxs = randi([1, size(T,1)], N_spls, 1);    % indices of samples to draw from param distributions

aSFK_res = zeros([size(sol.aSFK) N_spls]);
PG1_res = zeros([size(sol.PG1Stot) N_spls]);
PG1S_res = zeros([size(sol.PG1Stot) N_spls]);
rs = zeros(length(r), N_spls);    % initialize array of spatial grids from ensemble simulations
ts = zeros(length(t), N_spls);    % initialize array of time grids from ensemble simulations
count = 0;

% Run ensemble simulations:
for i = 1:length(idxs)
    D_i = Dmat(idxs(i),:);
    k_i = kmat(idxs(i),:);

    % Run simulations and save results:
    [sol_i,r_i,t_i] = basepdesolver(Co,D_i,k_i,R,dr,tf,Nts);    % spherical coordinates
%     [sol_i,r_i,t_i] = basepdesolver(Co,D_i,k_i,R,dr,tf,Nts);    % rectangular coordinates

    rs(:,i) = r_i;
    ts(:,i) = t_i;

    aSFK_res(:,:,i) = sol_i.aSFK;
    PG1_res(:,:,i) = sol_i.PG1tot;
    PG1S_res(:,:,i) = sol_i.PG1Stot;

    count = count + 1;
    disp(['Progress: ',num2str(count),'/',num2str(N_spls)])
end

%% Analyze and format ensemble results for plotting
% Calculate "median" surfaces:
aSFK_med = median(aSFK_res,3);
PG1_med = median(PG1_res,3);
PG1S_med = median(PG1S_res,3);

% Calculate upper and lower credible interal boundaries:
aSFK_up = quantile(aSFK_res, 0.5+0.341, 3);
aSFK_lo = quantile(aSFK_res, 0.5-0.341, 3);

PG1_up = quantile(PG1_res, 0.5+0.341, 3);
PG1_lo = quantile(PG1_res, 0.5-0.341, 3);

PG1S_up = quantile(PG1S_res, 0.5+0.341, 3);
PG1S_lo = quantile(PG1S_res, 0.5-0.341, 3);



%% Format results and generate variables for plotting
%Reduce number of nodes for plotting:
ns = 25; %number of plotting nodes
aSFK_plot2 = aSFK_med(1:Nr/ns:end,1:Nr/ns:end); %aSFKs
pGAB1_plot2 = PG1_med(1:Nr/ns:end,1:Nr/ns:end); %total pGAB1
pGAB1_SHP2_plot2 = PG1S_med(1:Nr/ns:end,1:Nr/ns:end); %total GAB1-SHP2
t2=t(1:Nr/ns:end); %new time vector for plotting 
x2=r(1:Nr/ns:end); %new distance vector for plotting



%% Plot results
close all
cmap = 'turbo';    %colormap
lw = 1.5;   %line width
fntsz = 12;
%aSFK surface plot
f1 = figure; colororder(julia_cols)
surf(t2,x2,aSFK_plot2, 'LineWidth', lw)
hold on
plot3(tf*ones(length(r)), r, aSFK_up(:,end), '--r')
plot3(tf*ones(length(r)), r, aSFK_lo(:,end), '--r')
set(gca, 'FontSize', fntsz, 'LineWidth', lw)
xlabel('min EGF')
ylabel('Distance from cell center (\mum)')
zlabel('aSFK (molec•µm^{-3})')
title('aSFK')
colormap(cmap)
xlim([0 tf])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%pGAB1 surface plot
f2 = figure; colororder(julia_cols)
surf(t2,x2,pGAB1_plot2, 'LineWidth', lw)
hold on
plot3(tf*ones(length(r)), r, PG1_up(:,end), '--r')
plot3(tf*ones(length(r)), r, PG1_lo(:,end), '--r')
set(gca,'FontSize', fntsz, 'LineWidth', lw)
xlabel('min EGF')
ylabel('r (μm)')
zlabel('pGAB1 (molec•µm^{-3})')
title('pGAB1')
colormap(cmap)
xlim([0 tf])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%GAB1-SHP2 surface plot
f3 = figure; colororder(julia_cols)
surf(t2,x2,pGAB1_SHP2_plot2, 'LineWidth', lw)
hold on
plot3(tf*ones(length(r)), r, PG1S_up(:,end), '--r')
plot3(tf*ones(length(r)), r, PG1S_lo(:,end), '--r')
set(gca,'FontSize', fntsz, 'LineWidth', lw)
xlabel('min EGF')
ylabel('r (μm)')
zlabel('GAB1-SHP2 (molec•µm^{-3})')
title('GAB1-SHP2')
colormap(cmap)
xlim([0 tf])