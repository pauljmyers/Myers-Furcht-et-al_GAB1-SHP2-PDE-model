## Required packages to run the functions below:
using Distributed
@everywhere begin
    using NumericalIntegration, ProgressMeter, SharedArrays
end 

# -- Define PDESystem and load necessary solver functions → may take several minutes to complete
@everywhere include("define_PDESystem_base.jl")  # change problem setup (e.g., r and t vectors) in this script  


## Functions:
@everywhere begin
    ## GSA functions -- diffusivities & kinetic params only:
    function pmap_fun_dk(p; 
        Co=Co, pvals=pvals, R=R, tf=tf, 
        pdesys=pdesys,  # model PDESystem
        prob=prob,      # ODEProblem of discretized PDESystem
        EGF=EGF, 
        D1=D1, D2=D2, D3=D3, D4=D4, D5=D5, D6=D6, D7=D7, 
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k7=k7, 
        k8=k8, k9=k9, k10=k10, k11=k11, k12=k12, k13=k13, k14=k14, 
        k15=k15, k16=k16, k17=k17,
        aSFK=aSFK, PG1S=PG1S, G2PG1S=G2PG1S, EG2PG1S=EG2PG1S,
        r=r, t=t,
        sa=1/surfCF, vol=1/volCF,
        rtol=1e-4
        )

        sol = solve(remake(prob, p=p, tspan=(0., tf)), QNDF(), saveat=tf, reltol=rtol, verbose=false)

        if sol.retcode == :Success
            r_sol = collect(sol[r])
            aSFK = sol(r_sol, tf, dv=aSFK(r,t))
    
            PG1S_cyt = sol(r_sol, tf, dv=PG1S(r,t)) .+ sol(r_sol, tf, dv=G2PG1S(r,t))   # cytoplasmic GAB1-SHP2
            PG1S_cyt_ave = NumericalIntegration.integrate(r_sol, PG1S_cyt.*r_sol.^2) .* 3.0 ./ R^3   # spatially averaged cytoplasmic GAB1-SHP2
            # PG1S_mem = sol(tf, dv=EG2PG1S(t)) .* sa/vol     # membrane GAB1-SHP2
            # PG1Stot = PG1S_cyt_ave .+ PG1S_mem     # total GAB1-SHP2
            
            # Length scale calculations:
            r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
            r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
            r12_pg1s = R - (r_sol[PG1S_cyt .>= 0.5.*maximum(PG1S_cyt)] |> minimum)    # r1/2, GAB1-SHP2
            r110_pg1s = R - (r_sol[PG1S_cyt .>= 0.1.*maximum(PG1S_cyt)] |> minimum)   # r1/10, GAB1-SHP2
            
            # Ratio of center-to-surface [PG1S]tot:
            cs_ratio = PG1S_cyt[1]/PG1S_cyt[end]
            
            return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1S_cyt_ave[1]]
        else
            return zeros(6)
        end
    end

    function fbatch_dk(p_batch; numout=6)
        p_in = [exp.(p_batch[:,i]) for i in axes(p_batch,2)]
        # p_in = [10.0 .^ (p_batch[:,i]) for i in axes(p_batch,2)]
        # p_in = [p_batch[:,i] for i in axes(p_batch,2)]
        pmap_out = @showprogress pmap(x->pmap_fun_dk(x), p_in;
            batch_size=100,
            on_error=zeros(numout)
            )
        out = hcat(pmap_out...)
        return out
    end

    function fbatch_dk_mt(p_batch; numout=6, Co=Co, pvals=pvals)
        # @show size(p_batch)
        p_batch = exp.(p_batch)
        # p_batch = 10.0 .^ (p_batch)
        out = zeros(numout, size(p_batch,2))
        prog = Progress(size(out,2))
        # for i in axes(out,2)
        Threads.@threads for i in axes(out,2)
            res_i = pmap_fun_dk(p_batch[:,i]; Co=Co, pvals=pvals) #try
            out[:,i] = res_i 
            
            # catch
            #     zeros(numout)
            # end
            # @show out[:,i]
            next!(prog)
        end
        return out
    end


    ## GSA functions for varying protein concentrations:
    # function pmap_fun_concs(p; Co=Co, pvals=pvals, R=R, dr=dr, tf=tf, tol=1e-3, maxiters=20)
    #     sol, r_sol = sapdesolver(p, D, kvals; R=R, dr=dr, tf=tf, tol=tol, maxiters=maxiters) # run model
    #     aSFK = sol.aSFK
    #     PG1Stot = sol.PG1Stot

    #     # Length scale calculations:
    #     r12_sfk = R - (r_sol[aSFK .>= 0.5.*maximum(aSFK)] |> minimum)   # r1/2, aSFK
    #     r110_sfk = R - (r_sol[aSFK .>= 0.1.*maximum(aSFK)] |> minimum)  # r1/10, aSFK
    #     r12_pg1s = R - (r_sol[PG1Stot .>= 0.5.*maximum(PG1Stot)] |> minimum)    # r1/2, GAB1-SHP2
    #     r110_pg1s = R - (r_sol[PG1Stot .>= 0.1.*maximum(PG1Stot)] |> minimum)   # r1/10, GAB1-SHP2

    #     # Ratio of center-to-surface [PG1S]tot:
    #     cs_ratio = PG1Stot[1]/PG1Stot[end]

    #     # Average [PG1S]tot:
    #     PG1Save = NumericalIntegration.integrate(r_sol, PG1Stot.*r_sol.^2).*3.0./R^3.0
    #     PG1Save_out = PG1Save[1]

    #     return [r12_sfk, r110_sfk, r12_pg1s, r110_pg1s, cs_ratio, PG1Save_out]
    # end

    # function fbatch_concs(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
    #     p_in = [p_batch[:,i] for i in axes(p_batch,2)]
    #     pmap_out = @showprogress pmap(pmap_fun_concs, p_in)
    #     out = hcat(pmap_out...)
    #     return out
    # end

    # function fbatch_concs_mt(p_batch; numout=6, Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
    #     # @show size(p_batch)
    #     p_batch = exp.(p_batch)
    #     # p_batch = 10.0 .^ (p_batch)
    #     out = zeros(numout, size(p_batch,2))
    #     prog = Progress(size(out,2))
    #     Threads.@threads for i in axes(out,2)
    #         out[:,i] = try
    #             pmap_fun_concs(p_batch[:,i]; Co=Co, D=Diffs, kvals=kvals, R=R, dr=dr, tf=tf)
    #         catch
    #             zeros(size(out[:,i]))
    #         end
    #         # @show out[:,i]
    #         next!(prog)
    #     end
    #     return out
    # end
end