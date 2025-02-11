using HDF5

function main()
    h5open("umschalt_eta_w_sweep.h5", "r") do file
        params_group = file["params"]

        # Load all parameters as a dictionary
        params_dict = Dict(Symbol(key) => read(params_group[key]) for key in keys(params_group))


        # Load sweep array
        sweep_array = read(file["sweep_array"])

        # Load solution data structure
        Omega = Dict(string(slice) => read(file["Omega/$(slice)"]) for slice in keys(file["Omega"]))
        rho_eg = Dict(string(slice) => read(file["rho_eg/$(slice)"]) for slice in keys(file["rho_eg"]))
        rho_ee = Dict(string(slice) => read(file["rho_ee/$(slice)"]) for slice in keys(file["rho_ee"]))
        rho_gg = Dict(string(slice) => read(file["rho_gg/$(slice)"]) for slice in keys(file["rho_gg"]))


        omega_200 = read(file["Omega/200"])
        println("Loaded Omega for slice 200, shape: ", size(omega_200))


        oneplotter(omega_200, params_dict)
        
    end


end

function oneplotter(sol, params_dict)


    #Timeshift, wie immer
    time = params_dict[:time]
    t_0 = params_dict[:t_0]
    geshift_t = time .- t_0


    Omega_plot = figure(figsize=(7, 5))
    om_ax = subplot(111)


    om_ax.plot(geshift_t, abs2.(sol[1,:]),lw=:1,color = "black", label = "slice 200" )

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

main()