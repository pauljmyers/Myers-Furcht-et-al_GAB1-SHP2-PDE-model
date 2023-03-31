#= 
This script defines functions for specifying parameter prior distributions using literature
estimates for the values and uncertainties of these parameters. These functions have been 
adapted for compatibility with Julia from the MATLAB codes provided by Tsigkinopoulou 
et al. in their Nature Protocols publication:

Tsigkinopoulou t al. Defining informative priors for ensemble modeling in systems biology. 
Nat Protoc 13, 2643–2663 (2018). https://doi.org/10.1038/s41596-018-0056-z.

IMPORTANT: Any use of these functions should include proper attribution and citation of the original
authors and their associated publication! *********************************
=#
using Distributions, StatsBase, Statistics, SpecialFunctions, NLsolve

"""
    This function generates the location and scale parameters for a lognormal
    distribution, given a mode, a Spread and a percentage of values that will 
    be contained within the defined boundaries. The function then generates the 
    location and scale parameters mu and sigma of the lognormal distribution, as well as  
    the Xminimum(=Mode/Spread) and Xmaximum(=Mode*Spread). The input can be given in two ways:\n 
    createLogNormDist(Mode, Spread) → only given Mode and  
    Spread. In this case, the percentage is set to the default value of 0.6827.\n
    createLogNormDist(Mode, Spread; Percentage)--> given Mode, Spread and percentage.\n
    The function then generates the location and scale parameters mu and sigma of the 
    lognormal distribution, as well as the Xminimum(=Mode/Spread) and Xmaximum(=Mode*Spread).
"""
function createLogNormDist(Mode, Spread; Percentage=0.6827)
    # Calculate Xmin and Xmax of the range of values from the Mode and Spread
    Xmin = Mode/Spread
    Xmax = Mode*Spread
    
    # Calculate mu and sigma of the lognormal distribution
    # @variables s
    # eqn = Percentage ~ 1/2+1/2*erf((log(Xmax)-(log(Mode)+s^2))/(sqrt(2)*s))-(1/2+1/2*erf((log(Xmin)-(log(Mode)+s^2))/(sqrt(2)*s)))
    # sigma = Symbolics.solve_for([eqn], [s])
    function f!(F, s)
        F[1] = Percentage - (1/2 + 1/2*erf.((log(Xmax) - (log(Mode) + s[1].^2))/(sqrt(2).*s[1])) - 
                    (1/2 + 1/2*erf.((log(Xmin) - (log(Mode) + s[1].^2))/(sqrt(2).*s[1]))))
    end

    σ = nlsolve(f!, [1.0], autodiff=:forward).zero[1]
    μ = log(Mode)+σ^2

    return μ, σ, Xmin, Xmax
end


##


"""
    This function calculates the weighted median from a vector of values based
    on their weights. The inputs are a vector D with the values and a vector W with
    their corresponding weights.
"""
function weightedMedian(D,W)
    if size(D) != size(W)
        error("weightedMedian: wrongMatrixDimension", "The dimensions of the input-matrices must match.")
    end
    
    # (line by line) transformation of the input-matrices to line-vectors
    d = reshape(permutedims(D),1,:)   
    w = reshape(permutedims(W),1,:)
    
    # sort the vectors
    A = [permutedims(d) permutedims(w)]
    A = A[sortperm(A[:,1]),:]
    ASort = A[.!any.(A[:,2] .≤ 1e-14), :]
    
    dSort = permutedims(ASort[:,1])
    wSort = permutedims(ASort[:,2])
    
    # If there is only one value, it is the weighted median
    if length(dSort)== 1
        wMed = dSort
    end
    
    # If there are only two values the weighted median is their mean if they 
    # have equal weights, otherwise the weighted median is the value with the
    # largest weight.
    if length(dSort) == 2
        if wSort[1] == wSort[2]
            wMed = [dSort[1] + dSort[2]]/2.0 
        elseif wSort[1] > wSort[2]
            wMed = dSort[1]
        else 
            wMed = dSort[2]
        end
    end
    
    # If there are more than two values, find the value which is the 50% weighted percentile:
    if length(dSort)!=1 && length(dSort)!=2
        i = 1
        j = length(wSort)
        Start = wSort[i]
        End = wSort[j]
        
        while i < j-1
            if Start-End > 1e-14
                End = End+wSort[j-1]
                j = j-1
            else
                Start = Start+wSort[i+1]
                i = i+1
            end
        end
        
        # If the 50% weighted percentile falls between two values, the weighted
        # median is the average of the two values if their individual weights are
        # equal, otherwise the weighted median is the value with the latgest
        # individual weight
        if abs(Start-End) < 1e-14
            wMed = (dSort[i]+dSort[j])/2.0
        elseif Start-End > 1e-13
            wMed = dSort[i]
        else
            wMed = dSort[j]
        end
    end
    return wMed
end 
    





"""
    This function generates the mode and confidence interval factor for a lognormal
    distribution, given a matrix P which contains 4 columns:\n
    1) The parameter values retrieved from literature \n
    2) The error of measurements in parameter values\n
    3) The weights of the parameter values\n
    4) A definition of the error as 
    multiplicative or additive. If the error is additive (Value +/- Error) input 0 
    and if it is multiplicative (Value */÷ Error) input 1 in the last column.\n
    Sample format of the table:\n
    V=[10000 NaN 2 0;100 0 8 0; 0.01 5 10 0; 160 NaN 4 0; 1 0.1 4 1]\n
    The input is given in the format:\n
    [Mode, CIfact] = MCIcalc(V)--> input table V.
"""
function calcModeSpread(V)
    D = Vector{Float64}()
    W = Vector{Float64}()
    
    lnE = Vector{Float64}(undef, size(V,1))
    lnP = Vector{Float64}(undef, size(V,1))
    
    for i = axes(V,1)
        if V[i,4] == 0 #log transform additive SD
           lnE[i] = sqrt(log(1+(V[i,2].^2 ./V[i,1].^2)))
           if isnan(V[i,2]) #if SD is NaN use 10% multiplicative SD
           lnP[i] = log(V[i,1])-1/2 .* log(1.1)^2
           else
           lnP[i] = log(V[i,1])-1/2 .* lnE[i].^2
           end
        else #log transform multiplicative SD
            lnP[i] = log(V[i,1])
            lnE[i] = log(V[i,2])
        end   
    end
    
    V[:,1] = lnP
    V[:,2] = lnE
    
    # Sort table from smallest to largest parameter value
    A = V[sortperm(V[:,1]), :]
    
    # Split table columns into separate vectors
    P = A[:,1]
    E = A[:,2]
    Wo = A[:,3]
    
    if any(Wo .< 0.0001)
        error("The weights cannot have values smaller than 0.0001.")
    end
    
    for i=eachindex(P)
        if isnan(E[i]) #if SD is NaN assign default 10% SD
            mu = P[i]  
            sigma = log(1.1)
            nbins = 1000; #generate bins within the parameter distribution range
            binEdges = collect(LinRange(mu-5*sigma,mu+5*sigma,nbins+1))
            aj = binEdges[1:end-1]
            bj = binEdges[2:end]
            cj = (aj + bj) ./ 2; #find centre of bins
            Pj = exp.(-(cj .- mu).^2 ./(2*sigma^2)) ./ (sigma*sqrt(2*pi))
            Wj = Wo[i]*Pj.*(bj - aj) #calculate weight at the centre of bins

        elseif E[i]!=0 #if SD is not NaN or 0 use logtransformed SD
            mu=P[i]
            sigma=E[i]
            nbins = 1000
            binEdges = collect(LinRange(mu-5*sigma,mu+5*sigma,nbins+1))
            aj = binEdges[1:end-1]
            bj = binEdges[2:end]
            cj = (aj .+ bj) ./ 2
            Pj = exp.(-(cj .- mu).^2 ./(2*sigma^2)) ./ (sigma*sqrt(2*pi))
            Wj=Wo[i]*Pj.*(bj .- aj)

        else 
            cj = P[i] #if SD is 0, do not assign SD but keep only the single value
            Wj = Wo[i]
        end
        
        #if the value is not the minimum, it's not a single value, and it does not overlap
        #with the previous value, generate additional bins between twice the distance of P[i] and 
        #P[i-1], otherwise do nothing   
         if P[i]!=minimum(P) && minimum(cj)>P[i-1] && length(cj)!=1 
              nbins2 = 1000
              binEdges2 = collect(LinRange(minimum(cj)-2*abs(minimum(cj)-P[i-1]), minimum(cj), nbins2+1))
              ajad = binEdges2[1:end-1]
              ajad=ajad[:]
              bjad = binEdges2[2:end]
              bjad=bjad[:]
              cjad = (ajad .+ bjad) ./ 2
              Pjad = exp.(-(cjad .- mu).^2 ./(2*sigma^2))./(sigma*sqrt(2*pi))
              Wjad=Wo[i]*Pjad.*(bjad .- ajad)
              
        else
           cjad = Vector{Float64}()
           Wjad = Vector{Float64}()
           
        end
    
        #if the value is not the maximum, it's not a single value, and it does not overlap
        #with the next value, generate additional bins between twice the distance of P[i] and 
        #P[i+1], otherwise do nothing       
        if P[i]!=maximum(P) && maximum(cj)<P[i+1] && length(cj)!=1
            nbins3 = 1000
            binEdges3 = collect(LinRange(maximum(cj), maximum(cj)+2*abs(P[i+1]-maximum(cj)), nbins3+1))
            ajad2 = binEdges3[1:end-1]
            bjad2 = binEdges3[2:end]
            cjad2 = (ajad2 .+ bjad2) ./ 2
            Pjad2 = exp.(-(cjad2 .- mu).^2 ./(2*sigma^2))./(sigma*sqrt(2*pi))
            Wjad2=Wo[i]*Pjad2.*(bjad2 .- ajad2)
        else 
            cjad2 = Vector{Float64}() 
            Wjad2 = Vector{Float64}()
            
        end
        
        append!(D, [cj; cjad; cjad2]) #add the centres of all bins in a vector
        append!(W, [Wj; Wjad; Wjad2]) #add the weights of all bin centres in a vector
    end
    wMed = weightedMedian(D, W) #calculate the weighted median of values in vector D
    
    S = Statistics.std(D, StatsBase.Weights(W)) #calculate the weighted standard deviation of values in vector D
    
    Mode = exp(wMed) #calculate Mode
    Spread = exp(S) #calculate Spread
    
    return Mode, Spread
end



"""
    The function generates a multivariate distribution for mass action reactions 
    with 3 parameters. The user must provide the mu and sigma for the three 
    log-normal distributions in the correct order, as well as the number of parameter 
    samples that are needed (N, default = 1000000).
"""
function multivariate3param(muKD, sigmaKD, mukon, sigmakon, mukoff, sigmakoff; N=1000000)
    
    # Define lognormal distributions:
    # KD 
    pd1 = LogNormal(muKD, sigmaKD)
    KD = rand(pd1, 1000000)
    
    #  kon 
    pd2 = LogNormal(mukon, sigmakon)
    kon = rand(pd2, 1000000)
    
    #  koff 
    pd3 = LogNormal(mukoff, sigmakoff)
    koff = rand(pd3, 1000000)
    
    #  Calculate coefficients of variation:
    
    GCV1 = exp(sigmaKD)-1
    GCV2 = exp(sigmakon)-1
    GCV3 = exp(sigmakoff)-1
    
    #  Choose the parameter with the largest coefficient of variation:
    A = [GCV1, GCV2, GCV3]
    M = maximum(A)
    
    #  Set dependent parameter based on the largest GCV and 
    #  calculate new mu and sigma for the dependent distribution:
    if M==GCV1 # KD dependent
       KD = koff./kon
       muKD = mukoff-mukon
       sigmaKD = sqrt(sigmakoff^2 + sigmakon^2)
    elseif M==GCV2 #  kon dependent
       kon = koff./KD; 
       mukon = mukoff-muKD
       sigmakon = sqrt(sigmakoff^2 + sigmaKD^2)
    elseif M==GCV3 #  koff dependent
       koff = kon.*KD
       mukoff = mukon+muKD
       sigmakoff = sqrt(sigmakon^2 + sigmaKD^2)
    end 
       
    if M==GCV2 #  kon dependent
       #  Generate multivariate distribution:
       println("MV parameter dist: kon and koff")
       B = [kon koff]
       CorrMat = Statistics.cor(B)
       μ = [mukon; mukoff]
       σ = [sigmakon; sigmakoff]
       
       #  Calculate the covariance structure
       σ_down = repeat(permutedims(σ), length(σ), 1)
       σ_acrs = repeat(σ, 1, length(σ))
       Σ = log.(CorrMat .* sqrt.(exp.(σ_down.^2.).-1.0) .* sqrt.(exp.(σ_acrs.^2.).-1.0) .+ 1.0)

       #  The Simulation
       mvdist = MvLogNormal(μ, Σ)
       y = permutedims(rand(mvdist, N))
       kon1 = y[:,1]
       koff1 = y[:,2]
       KD1 = koff1./kon1
       
    else #  koff or KD dependent
       #  Generate multivariate distribution:
       println("MV parameter dist: KD and koff")
       B = [KD koff]
       CorrMat = Statistics.cor(B)
       μ = [muKD; mukoff]
       σ = [sigmaKD; sigmakoff]

       #  Calculate the covariance structure
       σ_down = repeat(permutedims(σ), length(σ), 1)
       σ_acrs = repeat(σ, 1, length(σ))
       Σ = log.(CorrMat .* sqrt.(exp.(σ_down.^2).-1.0) .* sqrt.(exp.(σ_acrs.^2).-1.0) .+ 1.0)

       #  The Simulation
       mvdist = MvLogNormal(μ, Σ)
       y = permutedims(rand(mvdist, N))
       KD1 = y[:,1]
       koff1 = y[:,2]
       kon1 = koff1./KD1
    end
    return mvdist, KD1, kon1, koff1
end    