%% Diffusivity calculations for GAB1-SHP2 spatial model
clear
close all
clc

%Diffusivity conversion factor from cm^2/s --> um^2/min
cf = 1./(100)^2.*(1e6)^2.*60;

%Rh-tubulin
% D_rhtub = 1.61e-8*cf; %um^2/min
D_rhtub = mean(1e-8.*[1.61,1.34])*cf; %um^2/min --> using tubulin diffusivity values from Pepperkok et al. (DOI: 10.1083/jcb.111.6.3003) out to three decimal places
M_rhtub = 50000; %Diffusivity of tubulin, in Da

%Data from Erickson, 2009:
M = [14044; 25665; 42910; 69322; 157368; 239656; 489324; 606444];%; 387344]; %Molecular weights, in Da
Rs = [1.64; 2.09; 3.05; 3.55; 4.81; 5.20; 6.10; 8.50];%; 10.7]; %Stokes radii, in nm

%Perform linear regression with a cubic function
Aa = [M.^3, M.^2, M, ones(size(M))]; %Regression matrix
x = Aa\Rs;   %Correlation coefficient between protein molecular weight and Stokes radius


%Perform nonlinear regression with a logistic curve using Rs as independent
%variable:
Mlogfunc = @(Rs,A,B,C,K) K + (A-K)./(1 + exp(-B.*(Rs-C)));
objfunc = @(guess) sum((M-Mlogfunc(Rs,guess(1),guess(2),guess(3),guess(4))).^2);

options = optimset('MaxFunEvals',Inf','MaxIter',Inf,'TolFun',1e-4); %fminsearchbnd options

x0 = [6e5; 1; 1; min(M)];
lb = [-inf -inf -inf min(M)];
ub = [inf inf inf min(M)];
% [sol, fval] = fminsearchbnd(objfunc,x0,lb,ub,options)
% [sol, fval] = simulannealbnd(objfunc,x0,lb,ub,options)
[sol, fval] = lsqnonlin(objfunc, x0, lb, ub)

A = sol(1)
B = sol(2)
C = sol(3)
K = sol(4)
regvals = Mlogfunc(Rs,A,B,C,K)

%Plot regression
figure
plot(M,Rs,'.',M,Aa*x,'MarkerSize',20,'LineWidth',1)
xlabel('M (Da)')
ylabel('R (nm)')
title('Linear regression between protein mass and Stokes radius')
legend('Standards','Cubic linear regression','location','best')
set(gca,'LineWidth',1)

figure
plot(M,Rs,'.',Mlogfunc(linspace(min(Rs),max(Rs)),A,B,C,K),linspace(min(Rs),max(Rs)),'MarkerSize',20,'LineWidth',1)
xlabel('M (Da)')
ylabel('R (nm)')
title('Nonlinear regression between protein mass and Stokes radius')
legend('Standards','Correlation','location','best')
set(gca,'LineWidth',1)

Mi = [59835; 25206; 115000; 68436];    %Molecular weight estimates of SFKs, GRB2, GAB1, and SHP2

M_ms = [Mi; Mi(2)+Mi(3); sum(Mi(3:4));sum(Mi(2:end))]; %Molecular weights of all cytosolic model species
names = {'SFK';'GRB2';'GAB1';'SHP2';'GRB2-GAB1';'GAB1-SHP2';'GRB2-GAB1-SHP2'}; %Names associated with M_ms values

%% Estimate diffusivities with linear regression results
Rs_rhtub = [M_rhtub.^3 M_rhtub.^2 M_rhtub 1]*x;  %Estimated Stokes radius of Rh-tubulin
Rs_ms = [M_ms.^3 M_ms.^2 M_ms ones(size(M_ms))]*x;    %Estimated Stokes radii of model species

%Estimate diffusivity of model species i based on the ratio of Rs,tub/Rs,i
Rrats = Rs_rhtub./Rs_ms;    %Ratios of Rs,tub to model species Stokes radii

D_ms = D_rhtub.*Rrats;  %Diffusivities of model species based on diffusivity of tubulin

%Table of estimated diffusivities
tab = table(names,D_ms)

%% Estimate diffusivities using results from nonlinear regression
Rsfun = @(M) -1/B.*log((A-K)./(M-K)-1)+C;
Rs_ms2 = Rsfun(M_ms);
Rs_rhtub2 = Rsfun(M_rhtub);

%Estimate diffusivity of model species i based on the ratio of Rs,tub/Rs,i
Rrats2 = Rs_rhtub2./Rs_ms2;    %Ratios of Rs,tub to model species Stokes radii

D_ms2 = D_rhtub.*Rrats2;  %Diffusivities of model species based on diffusivity of tubulin

%Table of estimated diffusivities
tab2 = table(names,D_ms2)

%Plot regression
figure
plot(M,Rs,'.',M_ms,Rs_ms2,'.','MarkerSize',20','LineWidth',1)
xlabel('M (Da)')
ylabel('R (nm)')
legend('Standards','Regression','location','best')

%% Calculate diffusivities using a cubic spline interpolation
%Define molecular weight range
Mint = linspace(min(M),max(M));
%Calculate piecewise cubic hermite interpolating polynomial spline over desired molecular weight range using standards
Rsint = pchip(M,Rs,Mint);

%Get polynomial coefficients for cubic spline above
Rspp = pchip(M,Rs);
%Get spline values for Rs at molecular weights of model species
Rspp_ms = ppval(Rspp,M_ms);
Rspp_tub = ppval(Rspp,M_rhtub);

%Plot interpolation
figure
plot(M,Rs,'.',M_ms,Rspp_ms,'.',Mint,Rsint,'MarkerSize',20,'LineWidth',1)
xlabel('M (Da)')
ylabel('R (nm)')
title('Correlation between protein mass and Stokes radius')
legend('Standards','Spline','location','best')
set(gca,'LineWidth',1)

%Calculate diffusivities based off diffusivity of tubulin
Rrats3 = Rspp_tub./Rspp_ms;    %Ratios of Rs,tub to model species Stokes radii

D_ms3 = D_rhtub.*Rrats3;  %Diffusivities of model species based on diffusivity of tubulin

%Table of estimated diffusivities
tab3 = table(names,D_ms3)

%% Calculate diffusivities using a linear interpolation **** --> this is the one that gets the numbers Chris has
%Calculate linear interpolation for model species and using standards
Rsvq = interp1(M,Rs,M_ms);
%Calculate the same for tubulin
Rsvqtub = interp1(M,Rs,M_rhtub);
RsvqSOS = interp1(M,Rs,116403+115000);

%Plot interpolation
figure
plot(M,Rs,'.',M_ms,Rsvq,'.','MarkerSize',20,'LineWidth',1)
xlabel('M (Da)')
ylabel('R (nm)')
title('Correlation between protein mass and Stokes radius')
legend('Standards','Interpolation prediciton','location','best')
set(gca,'LineWidth',1)

%Calculate diffusivities based off diffusivity of tubulin
Rrats4 = Rsvqtub./Rsvq;    %Ratios of Rs,tub to model species Stokes radii
RratSOS = Rsvqtub./RsvqSOS; %" " to Stokes radius of SOS

D_sos = D_rhtub.*RratSOS;
D_ms4 = D_rhtub.*Rrats4;  %Diffusivities of model species based on diffusivity of tubulin

%Table of estimated diffusivities
tab4 = table(names,D_ms4)

