using DifferentialEquations
using Plots
using LinearAlgebra
using SpecialFunctions
using NaNMath
using LaTeXStrings
using Measures
using ProgressLogging
using JLD2
using PyPlot
#using Profile
using StaticArrays
#using BenchmarkTools


function EOM!(dt, y, p, t)
    #----------------params---------------------------
    (t_0, u, w, N, Gamma, Delta, eta, zmax, Amp,h) = p

    rho_gg = @view y[1:N]
    rho_ee = @view y[(N+1):(2*N)]
    rho_eg = @view y[(2*N)+1:(3*N)]
    Omega  = @view y[(3*N)+1:(4*N)]

    vierte_1 = @SVector[-1/4, -5/6, 3/2, -1/2, 1/12]
    vierte_i = @SVector[1/12, -2/3, 0, 2/3, -1/12]
    vierte_N_1 = @SVector[-1/12, 1/2, -3/2, 5/6, 1/4]
    vierte_N = @SVector[1/4, -4/3, 3, -4, 25/12]

    #----------rho_gg, rho_ee, rho_eg, in this order-----------
    @inbounds for i in 1:N
        real_part = real(Omega[i]) * real(rho_eg[i]) + imag(Omega[i]) * imag(rho_eg[i])
        imag_part = imag(Omega[i]) * real(rho_eg[i]) - real(Omega[i]) * imag(rho_eg[i])
    
        dt[i]       = Gamma * rho_ee[i] + imag_part
        dt[i + N]   = -Gamma * rho_ee[i] - imag_part
        dt[i + 2*N] = (-Gamma / 2 + 1im * Delta) * rho_eg[i] + (1im / 2) * Omega[i] * (rho_gg[i] - rho_ee[i])
    end

    #-------------------------first slice--------------------    
        #Abschneidung von Gauss fuer einfachere numerische Kalkulation
    erste = -2 * Amp / w^2 * (t - t_0) * exp(-((t - t_0)^2) / w^2)

    if abs(t - t_0) <= 5 * w
        dt[(3*N)+1] = erste 
    else
        dt[(3*N)+1] = 0
    end
#------------------Omegas--------------------------
    for i in 2:N
        if i==2
            dt[i+3*N] = ((-1/(u * h)) * (vierte_1[1] * Omega[i-1] + vierte_1[2] * Omega[i]+ 
            vierte_1[3] * Omega[i+1]+ vierte_1[4] * Omega[i+2]+ vierte_1[5] * 
            Omega[i+3]) + (1im * eta) / u * rho_eg[i] )
        elseif 3 <= i < N-1
            dt[i+3*N] = ((-1/(u * h)) * (vierte_i[1] * Omega[i-2] + vierte_i[2] * Omega[i-1]+ 
            vierte_i[3] * Omega[i]+ vierte_i[4] * Omega[i+1]+ vierte_i[5] * 
            Omega[i+2]) + (1im * eta) / u * rho_eg[i] )
        elseif i == N-1
            dt[i+3*N] = ((-1/(u * h)) * (vierte_N_1[1] * Omega[i-3] + vierte_N_1[2] * Omega[i-2]+ 
            vierte_N_1[3] * Omega[i-1]+ vierte_N_1[4] * Omega[i]+ vierte_N_1[5] * 
            Omega[i+1]) + (1im * eta) / u * rho_eg[i] )
        elseif i == N
            dt[i+3*N] = ((-1/(u * h)) * (vierte_N[1] * Omega[i-4] + vierte_N[2] * Omega[i-3]+ 
            vierte_N[3] * Omega[i-2]+ vierte_N[4] * Omega[i-1]+ vierte_N[5] * 
            Omega[i]) + (1im * eta) / u * rho_eg[i] )
        end
    end


end #funcEOM!


function EOM_solver(params)
    t_start = params[:t_start]
    t_end = params[:t_end]
    t_step = params[:t_step]
    t_0 = params[:t_0]
    u = params[:u]
    w = params[:w]
    N = params[:N]
    Gamma = params[:Gamma]
    Delta = params[:Delta]
    eta = params[:eta]
    zmax = params[:zmax]
    Amp = params[:Amp]

    # Initial conditions
    dt_0 = ones(4*N) .+ 0im
    dt_0[N+1:end] .= 0 
    
    h = zmax / N
    # EOM parameters
    p = (t_0, u, w, N, Gamma, Delta, eta, zmax, Amp,h)
    tspan =(t_start, t_end)

    prob = ODEProblem(EOM!, dt_0, tspan, p)
    @time sol = solve(prob, dtmax = t_step, progress = true,saveat=0.01)

    rho_gg = sol[1:N, :]        #rho_gg
    rho_ee = sol[N+1:2*N, :]    #rho_ee
    rho_eg= sol[2*N+1:3*N, :]  #rho_eg
    Omega = sol[3*N+1:4*N, :]  #Omega_sol
    
    time = sol.t         #time

    params[:time] = time

    return Omega, rho_eg, rho_ee, rho_gg
end


function theo_resp(params, time, plotslice)
    u = params[:u]
    Gamma = params[:Gamma]
    Delta = params[:Delta]
    eta = params[:eta]
    zmax = params[:zmax]
    N = params[:N]
    t_0 = params[:t_0]

    plot_zmax = (plotslice /N) * zmax

    b = eta * plot_zmax / 2
    y = zeros(size(time)) .+ 0im
    
    for (j, t) in enumerate(time)
        if t == 0
            y[j]=0
        else
            y[j] = @.  - sqrt((b / t)) * besselj1(2 * sqrt(b * t)) * exp(- Gamma * t / 2) * exp(1im * Delta * t) 
        end
    end
    return y
end

function idx_finder(arr::AbstractArray{T}, value) where T
    idx = argmin(abs.(arr .- value))
    #print(abs.(arr .- value))
    return idx
end

function unwrap!(x, period = 2π)
    y = convert(eltype(x), period)
    v = first(x)
    @inbounds for k = eachindex(x)
        x[k] = v = v + rem(x[k] - v,  y, RoundNearest)
    end
    return x
end

#__________________________________________________________________________________________________________________
 

function onesolver()
    params = Dict(
        :t_start => 0,
        :t_end => 12,   #keep in mind, its actually t_end - t_0 displayed
        :t_0 => 4,
        :u => 6/10,
        :w => 1 / 45,
        :N => 500,
        :Gamma => 1,
        :Delta => 0,
        :eta => 8,
        :zmax => 2,
        :t_step => 1e-3,
        :Amp => 2,
        :time => zeros(100)
    )


    sol_EOM = EOM_solver(params)

    return sol_EOM, params
end # solution



function oneplotter(sol, params)

    Omega, rho_eg, rho_ee, rho_gg = sol
    print("\n plotter:basic plotter  ")
    print("| Size of Omega:  ", size(Omega), "  |")

    #Timeshift, wie immer
    time = params[:time]
    t_0 = params[:t_0]
    geshift_t = time .- t_0

    #Normierung, bei dem Wert norm_x
    norm_x = 1
    norm_idx = idx_finder(geshift_t, norm_x )

    select_slice = 300
    # #X_OFFSet:
    # u = params[:u]
    # slices = params[:slices]
    # z_max = params[:zmax]
    # N = params[:N]
    # x_offset = u * slices[select_slice] * z_max / N


    Omega_plot = figure(figsize=(7, 5))
    om_ax = subplot(111)


    om_ax.plot(geshift_t, abs2.(Omega[select_slice,:]),lw=:1,color = "black", label = "slice $select_slice" )

    om_ax.set_title(latexstring("| \\Omega |^2 "))
    om_ax.set_ylabel("Intensity [a.u.]")
    om_ax.set_xlabel(latexstring("\\mathrm{Time} \\; \\Gamma t"))
    om_ax.set_ylim(1e-15, 100)
    om_ax.set_xlim(0,8)
    om_ax.set_yscale("log")
    om_ax.legend(loc = "upper right")
    


    display(Omega_plot)
    #savefig("Desktop/report_29.04/1.2.highexc.svg")


end





function main()
    

    sol, params = onesolver()
    oneplotter(sol, params)


    matplotlib.pyplot.close()
end

function test_main()

    onesolver()
    print("done")

end


main()
