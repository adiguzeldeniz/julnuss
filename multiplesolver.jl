
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
using StaticArrays
using HDF5

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



function EOM_twoblocks_solver(params)
    t_start = params[:t_start]
    t_end = params[:t_end]
    t_stepA = params[:t_stepA]
    t_stepB = params[:t_stepB]
    t_jumpAB = params[:t_jumpAB]
    t_0 = params[:t_0]
    u = params[:u]
    w = params[:w]
    N = params[:N]
    Gamma = params[:Gamma]
    Delta = params[:Delta]
    eta_A = params[:eta_A]
    eta_B = params[:eta_B]
    zmax = params[:zmax]
    Amp = params[:Amp]

    h = zmax / N

    function solve_block(tspan,t_step, p, init_cond)
        prob = ODEProblem(EOM!, init_cond, tspan, p)
        @time sol = solve(prob, dtmax = t_step, progress = true)
        return sol
    end

    #______________________|BLOCK|A|______________________
    dt_0A = ones(4*N) .+ 0im
    dt_0A[N+1:end] .= 0 
    pA= (t_0, u, w, N, Gamma, Delta, eta_A, zmax, Amp,h)
    tspanA =(t_start, t_jumpAB)
    solA = solve_block(tspanA,t_stepA, pA, dt_0A)

    #______________________|BLOCK|B|______________________
    dt_0B = [solA[i, end] for i in 1:(4 * N)]
    pB= (t_0, u, w, N, Gamma, Delta, eta_B, zmax, Amp,h)
    tspanB = (t_jumpAB, t_end)
    solB = solve_block(tspanB,t_stepB, pB, dt_0B)


    function combine_results(solA, solB, N)
        time = vcat(solA.t, solB.t)
        Omega = hcat(solA[3*N+1:4*N, :], solB[3*N+1:4*N, :])
        rho_eg = hcat(solA[2*N+1:3*N, :], solB[2*N+1:3*N, :])
        rho_ee = hcat(solA[N+1:2*N, :], solB[N+1:2*N, :])
        rho_gg = hcat(solA[1:N, :], solB[1:N, :])
        return (time, Omega, rho_eg, rho_ee, rho_gg)
    end

    time_ges, Omega_ges, rhoeg_ges, rhoee_ges, rhogg_ges = combine_results(solA, solB, N)
    params[:time] = time_ges
    return Omega_ges, rhoeg_ges, rhoee_ges, rhogg_ges
end

#__________________________________________________________________________________________________________________

function multiplesolverandsaver_hdf5()
    params = Dict(
        :t_start => 0,
        :t_end => 12,
        :t_0 => 4,
        :u => 85/100,
        :w => 1 / 45,
        :N => 600,
        :Gamma => 1,
        :Delta => 0,
        :eta_A => 8,
        :eta_B => 8,
        :zmax => 2,
        :Amp => 2,
        :time => zeros(100),  # Reference array, but actual time steps are determined dynamically
        :slices => [5, 200, 400],  # Selected spatial slices
        :t_stepA => 1e-3,
        :t_stepB => 1e-3,
        :t_jumpAB => 1,
        :sweep_array => zeros(20)
    )

    params[:t_jumpAB] = params[:t_0] + params[:u] * params[:zmax]

    #________________select slices to save________________
    params[:slices] = [5,200,400]

    # Define sweep array
    par_start, par_end, par_step = 0.001, 4, 0.5
    sweep_array = collect(par_start:par_step:par_end)
    num_sweeps = length(sweep_array)

    params[:eta_A] = params[:eta_B] * 0.2

    # Run one sample solution to determine actual time steps
    sample_sol = EOM_twoblocks_solver(params)
    num_time_steps = size(sample_sol[1], 2)  # Extract the correct number of time steps

    # Open HDF5 file
    file_name = "umschalt_eta_w_sweep.h5"
    h5open(file_name, "w") do file
        # Save the full sweep array
        file["sweep_array"] = sweep_array

        # Create a dedicated group for parameters
        params_group = create_group(file, "params")

        # Save all parameters as datasets (instead of attributes)
        for (key, value) in params
            params_group[string(key)] = value  # Store everything as datasets
        end

        # Create hierarchical structure
        for group_name in ["Omega", "rho_eg", "rho_ee", "rho_gg"]
            g = create_group(file, group_name)
            for slice_index in params[:slices]
                create_dataset(g, string(slice_index), ComplexF64, num_sweeps, num_time_steps)
            end
        end

        for (j, sweep_fact) in enumerate(sweep_array)
            println("Solving for sweep factor: $sweep_fact ($j / $num_sweeps)")
            params[:t_jumpAB] = params[:t_0] + params[:u] * params[:zmax] * sweep_fact

            sol = EOM_twoblocks_solver(params)

            #println("Sample Omega Before Saving: ", sol[1][400:405, 5000:5005])

            # Validate slice indices
            #slice_indices = filter(i -> i ≤ size(sol[1], 1), params[:slices])
            #if isempty(slice_indices)
            #    error("All slice indices are out of bounds!")
            #end

            for slice_index in params[:slices]
                file["Omega/$(slice_index)"][j, :] = sol[1][slice_index, :]
                file["rho_eg/$(slice_index)"][j, :] = sol[2][slice_index, :]
                file["rho_ee/$(slice_index)"][j, :] = sol[3][slice_index, :]
                file["rho_gg/$(slice_index)"][j, :] = sol[4][slice_index, :]
                flush(file)  # Ensure data is committed
            end

        end
    end

    println("All solutions saved to $file_name")
end



function main()
    multiplesolverandsaver_hdf5()
end

main()