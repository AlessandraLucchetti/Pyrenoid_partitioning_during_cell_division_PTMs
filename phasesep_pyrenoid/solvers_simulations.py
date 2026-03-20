import phasesep_pyrenoid as ph
import numpy as np
import pde
# Dictionary of initialization methods
INITIAL_CONDITIONS = {
    "from_file": ph.init_from_file,
    "random": ph.init_random,
    "single_droplet": ph.init_single_droplet,
    "emulsion": ph.init_emulsion
}


def solve_stokes_flow(nx,ny, Lx, Ly, mu, eta, P0, sigma_P):
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Grid spacing

    # Create mesh grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    P = P0 * np.exp(-(X - Lx / 2)**2 / (2*sigma_P**2)) #gaussian pressure profile
    dpdx = -P0 * (X - Lx / 2) / (sigma_P**2) * np.exp(-(X - Lx / 2)**2 / (2*sigma_P**2)) 
    tolerance = 1e-6  # Convergence criteria

    # Initialize velocity field
    u = np.zeros((nx, ny))  # Velocity in x-direction
    u_new = np.zeros_like(u)

    # Boundary conditions
    u[:, 0] = u[:, 1] 
    u[:, -1] = u[:, -2]  
    u[0, :] = u[1,:] 
    u[-1, :] = u[-2,:] 

    # Iterative solver (Gauss-Seidel)
    error = 1
    while error > tolerance:
        u_new[:] = u[:]

        u[1:-1, 1:-1] = (
            (u[2:, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] + u[1:-1, :-2]) / dy**2 -
            dpdx[1:-1, 1:-1] / mu
        ) / (2 / dx**2 + 2 / dy**2 + eta/mu)

        # Apply boundary conditions
        u[:, 0] = u[:, 1] 
        u[:, -1] = u[:, -2]  
        u[0, :] = u[1,:] 
        u[-1, :] = u[-2,:] 

        # Compute error
        error = np.max(np.abs(u - u_new))
    return P, u, x, y, X, Y

def run_simulation(

    selected_init_cond_U,
    selected_init_cond_M,
    PARAMS_INIT_U,
    PARAMS_INIT_M,
    save_final_config = False,
    input_filename = None,
    output_filename = None,
    output_folder = None,
    chem_react: bool = False,
    chem_react_type = "None",
    k_M: float = 0.01,
    k_U: float = 0.01,
    xc = 0.5,
    b = 200,
    d: float = 0.5, # diffusivity ratio
    kappa: float = 1,
    chi: float = -4.0, # interaction parameter
    u = None,
    periodic_bc : bool = False,
    shape: float = 100, # length of one axis in space
    size: float = 200, # number  of support points along each axis
    simulation_time: float = 15,
    interval_for_tracking: float = 3,
    advection: bool = False,
    cmapcolor = "RdBu"

):
    SEED = 8
    rng = np.random.default_rng(SEED)

    GRID = pde.CartesianGrid([[0, shape], [0, shape]], [size, size], periodic=periodic_bc)

    if selected_init_cond_U == "from_file":
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](input_filename, PARAMS_INIT_U)
    else:
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](GRID, rng, PARAMS_INIT_U)

    if selected_init_cond_M == "from_file":
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](input_filename, PARAMS_INIT_M)
    else:
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](GRID, rng, PARAMS_INIT_M)

    STATE = pde.FieldCollection([fieldU, fieldM])         # Create field collection

    # Save parameters
    ph.save_parameters(PARAMS_INIT_U, PARAMS_INIT_M, output_folder+"init_cond_params.json")

    # Plot
    STATE.plot(title = "Initial Phase Field", clim = [0,1], cmap = cmapcolor, colorbar = False)#, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])


    if chem_react:
        if chem_react_type == "int_maint":
            reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]+{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
            reactionb = ph.Reaction([0, 1, 0], f"{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]-{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
        elif chem_react_type == "ext_maint":
            reactiona = ph.Reaction([1, 0, 0], f"-{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]+{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
            reactionb = ph.Reaction([0, 1, 0], f"{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]-{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
            
        reactions = ph.Reactions(3, [reactiona, reactionb])

        # if chem_react_type == "int_maint":
        #     reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*c[0]+{k_U}*c[1]", conservative=False)
        #     reactionb = ph.Reaction([0, 1, 0], f"{k_M}*c[0]-{k_U}*c[1]", conservative=False)
        # elif chem_react_type == "ext_maint":
        #     reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*c[0]+{k_U}*c[1]*(1-c[0])", conservative=False)
        #     reactionb = ph.Reaction([0, 1, 0], f"{k_M}*c[0]-{k_U}*c[1]*(1-c[0])", conservative=False)
            
        reactions = ph.Reactions(3, [reactiona, reactionb])

    else:
        reactions = None

    # set interaction matrix
    chi_m = np.zeros((3, 3))
    chi_m[0, 2] = chi_m[2, 0] = chi
    kappa_m = kappa * np.diag([1.0, 0.0, 0.0])     # set kappa matrix

    # O: U, 1: M, 2: S

    x, y = GRID.axes_coords
    Y,X = np.meshgrid(x,y)
    f = ph.FloryHugginsNComponents(3, chis=chi_m) #, variables=["a", "b"])

    if advection:      
        velocity_field = np.ones((2, size+2, size+2))
        velocity_field[0,:,:] = u
        velocity_field[1,:,:] = 0
    else:
        velocity_field = None
    
    # define Cahn Hilliard PDE including chemical reactions
    eq = ph.CahnHilliardMultiplePDE(
        {
            "free_energy": f,
            "kappa": kappa_m,
            "mobility": [d, 1, 1], # I think the third value is solvent diffusivity but it is ignored since phi_S = 1 - sum(other phis) so it should not matter?
            "reactions": reactions,
            "velocity_field": velocity_field,
        }
    )

    storage_write = pde.FileStorage(output_filename+".hdf")
    # set trackers
    storage = pde.MemoryStorage()


    trackers = [
        "progress",                                    # show progress bar during simulation
        storage.tracker(interval=interval_for_tracking),            # store data every interval
        storage_write.tracker(interrupts=interval_for_tracking),    
        #pde.PlotTracker(interval="0:10"),             # show images during simulation
        "steady_state",                                # abort when steady state is reached
    ]

    # run simulation
    sol, solver_info = eq.solve(
        STATE,
        simulation_time,
        solver = "explicit", # Euler is default.  Can also be "implicit". 
        #maxerror = 0.001,
        #maxiter = 10,
        dt = 0.0001,           # If adaptive = True, this will only be the initial time step and then it will be adapted
        tolerance=1e-4, #6,
        adaptive=True,
        tracker=trackers,
        ret_info = True
    )

    # save movie

    pde.movie(
        storage,
        f"{output_folder}movie.mp4",
        plot_args={"vmin": 0, "vmax": 1, "cmap":cmapcolor, "subplot_args": [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}]})
    
    # return final state
    solution = {
        "sol0": sol[0].data,
        "sol1": sol[1].data,
    }

    if save_final_config:
        storage_read = pde.FileStorage(output_filename+".hdf")
        for time, UM_fields in storage_read.items():
            if time == simulation_time:
                    U_field = UM_fields[0]
                    U_field.to_file(output_filename+"_U.hdf5")
                    M_field = UM_fields[1]
                    M_field.to_file(output_filename+"_M.hdf5")
                    
    sol.plot(title = "Final Phase Field", colorbar = True, clim = [0,1], cmap = cmapcolor, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])

    return solution, solver_info

def run_simulation_resubmit(

    selected_init_cond_U,
    selected_init_cond_M,
    PARAMS_INIT_U,
    PARAMS_INIT_M,
    save_final_config = False,
    input_filename = None,
    output_filename = None,
    output_folder = None,
    chem_react: bool = False,
    chem_react_type = "None",
    k_M: float = 0.01,
    k_U: float = 0.01,
    xc = 0.5,
    b = 200,
    d: float = 0.5, # diffusivity ratio
    kappa: float = 1,
    chi: float = -4.0, # interaction parameter
    u = None,
    periodic_bc : bool = False,
    shape: float = 100, # length of one axis in space
    size: float = 200, # number  of support points along each axis
    simulation_time: float = 15,
    interval_for_tracking: float = 3,
    advection: bool = False,
    cmapcolor = "RdBu"

):
    SEED = 8
    rng = np.random.default_rng(SEED)

    GRID = pde.CartesianGrid([[0, shape], [0, shape]], [size, size], periodic=periodic_bc)

    if selected_init_cond_U == "from_file":
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](input_filename, PARAMS_INIT_U)
    else:
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](GRID, rng, PARAMS_INIT_U)

    if selected_init_cond_M == "from_file":
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](input_filename, PARAMS_INIT_M)
    else:
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](GRID, rng, PARAMS_INIT_M)

    STATE = pde.FieldCollection([fieldU, fieldM])         # Create field collection

    # Save parameters
    ph.save_parameters(PARAMS_INIT_U, PARAMS_INIT_M, output_folder+"init_cond_params.json")

    # Plot
    STATE.plot(title = "Initial Phase Field", clim = [0,1], cmap = cmapcolor, colorbar = False)#, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])


    if chem_react:
        # if chem_react_type == "int_maint":
        #     reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]+{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
        #     reactionb = ph.Reaction([0, 1, 0], f"{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]-{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
        # elif chem_react_type == "ext_maint":
        #     reactiona = ph.Reaction([1, 0, 0], f"-{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]+{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
        #     reactionb = ph.Reaction([0, 1, 0], f"{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]-{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
            
        #reactions = ph.Reactions(3, [reactiona, reactionb])

        if chem_react_type == "int_maint":
            reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*c[0]+{k_U}*c[1]", conservative=False)
            reactionb = ph.Reaction([0, 1, 0], f"{k_M}*c[0]-{k_U}*c[1]", conservative=False)
        elif chem_react_type == "ext_maint":
            reactiona = ph.Reaction([1, 0, 0], f"-{k_M}*c[0]+{k_U}*c[1]*(1-c[0])", conservative=False)
            reactionb = ph.Reaction([0, 1, 0], f"{k_M}*c[0]-{k_U}*c[1]*(1-c[0])", conservative=False)
            
        reactions = ph.Reactions(3, [reactiona, reactionb])

    else:
        reactions = None

    # set interaction matrix
    chi_m = np.zeros((3, 3))
    chi_m[0, 2] = chi_m[2, 0] = chi
    chi_m[1, 0] = chi_m[0,1] = chi
    kappa_m = kappa * np.diag([1.0, 1.0, 1.0])     # set kappa matrix

    # O: U, 1: M, 2: S

    x, y = GRID.axes_coords
    Y,X = np.meshgrid(x,y)


    a = 0.5 #if t>10 else 0
    xc1 = 20
    xc2 = 25
    yc = 20
    v1 = -0.05 
    v2 = 0.05
    # tstop = 50
    # t0 = tstop # if t < tstop else tstop
    c1 = 3#3
    c2 = 3
    func =  a * np.exp(-((X-xc1)**2/(2*c1**2))) #t/simT * 2 * a * np.exp(-(X-xc)**2/(2*c**2))
    external_potential_arr = np.stack((func, func, 0*func))

    f = ph.FloryHugginsNComponentsExternalPotential(3, chis=chi_m, x_grid_matrix=X, y_grid_matrix=Y, external_potential=external_potential_arr) #, variables=["a", "b"])

    if advection:      
        velocity_field = np.ones((2, size+2, size+2))
        velocity_field[0,:,:] = u
        velocity_field[1,:,:] = 0
    else:
        velocity_field = None
    
    # define Cahn Hilliard PDE including chemical reactions
    eq = ph.CahnHilliardMultiplePDE(
        {
            "free_energy": f,
            "kappa": kappa_m,
            "mobility": [d, 1, 1], # I think the third value is solvent diffusivity but it is ignored since phi_S = 1 - sum(other phis) so it should not matter?
            "reactions": reactions,
            "velocity_field": velocity_field,
        }
    )

    storage_write = pde.FileStorage(output_filename+".hdf")
    # set trackers
    storage = pde.MemoryStorage()


    trackers = [
        "progress",                                    # show progress bar during simulation
        storage.tracker(interval=interval_for_tracking),            # store data every interval
        storage_write.tracker(interrupts=interval_for_tracking),    
        #pde.PlotTracker(interval="0:10"),             # show images during simulation
        "steady_state",                                # abort when steady state is reached
    ]

    # run simulation
    sol, solver_info = eq.solve(
        STATE,
        simulation_time,
        solver = "explicit", # Euler is default.  Can also be "implicit". 
        #maxerror = 0.001,
        #maxiter = 10,
        dt = 0.0001,           # If adaptive = True, this will only be the initial time step and then it will be adapted
        tolerance=1e-4, #6,
        adaptive=True,
        tracker=trackers,
        ret_info = True
    )

    # save movie

    pde.movie(
        storage,
        f"{output_folder}movie.mp4",
        plot_args={"vmin": 0, "vmax": 1, "cmap":cmapcolor, "subplot_args": [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}]})
    
    # return final state
    solution = {
        "sol0": sol[0].data,
        "sol1": sol[1].data,
    }

    if save_final_config:
        storage_read = pde.FileStorage(output_filename+".hdf")
        for time, UM_fields in storage_read.items():
            if time == simulation_time:
                    U_field = UM_fields[0]
                    U_field.to_file(output_filename+"_U.hdf5")
                    M_field = UM_fields[1]
                    M_field.to_file(output_filename+"_M.hdf5")
                    
    sol.plot(title = "Final Phase Field", colorbar = True, clim = [0,1], cmap = cmapcolor, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])

    return solution, solver_info


def run_simulation_binary(

    selected_init_cond_U,
    selected_init_cond_M,
    PARAMS_INIT_U,
    PARAMS_INIT_M,
    save_final_config = False,
    input_filename = None,
    output_filename = None,
    output_folder = None,
    chem_react: bool = False,
    chem_react_type = "None",
    k_M: float = 0.01,
    k_U: float = 0.01,
    xc = 0.5,
    b = 200,
    d: float = 0.5, # diffusivity ratio
    kappa: float = 1,
    chi: float = -4.0, # interaction parameter
    u = None,
    periodic_bc : bool = False,
    shape: float = 100, # length of one axis in space
    size: float = 200, # number  of support points along each axis
    simulation_time: float = 15,
    interval_for_tracking: float = 3,
    advection: bool = False,
    cmapcolor = "RdBu"

):
    SEED = 8
    rng = np.random.default_rng(SEED)

    GRID = pde.CartesianGrid([[0, shape], [0, shape]], [size, size], periodic=periodic_bc)

    if selected_init_cond_U == "from_file":
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](input_filename, PARAMS_INIT_U)
    else:
        fieldU = INITIAL_CONDITIONS[selected_init_cond_U](GRID, rng, PARAMS_INIT_U)

    if selected_init_cond_M == "from_file":
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](input_filename, PARAMS_INIT_M)
    else:
        fieldM = INITIAL_CONDITIONS[selected_init_cond_M](GRID, rng, PARAMS_INIT_M)

    STATE = pde.FieldCollection([fieldU])         # Create field collection

    # Save parameters
    ph.save_parameters(PARAMS_INIT_U, PARAMS_INIT_M, output_folder+"init_cond_params.json")

    # Plot
    STATE.plot(title = "Initial Phase Field", clim = [0,1], cmap = cmapcolor, colorbar = False)#, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])


    if chem_react:
        if chem_react_type == "int_maint":
            reactiona = ph.Reaction([1, 0], f"-{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]+{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
            reactionb = ph.Reaction([0, 1], f"{k_M}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[0]-{k_U}*(1/(1+2.718**(-{b}*(c[0]-{xc}))))*c[1]", conservative=False)
        elif chem_react_type == "ext_maint":
            reactiona = ph.Reaction([1, 0], f"-{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]+{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*(1-c[0])", conservative=False)
            reactionb = ph.Reaction([0, 1], f"{k_M}/(1+2.718**(-{b}*(c[0]-{xc})))*c[0]-{k_U}*(1-1/(1+2.718**(-{b}*(c[0]-{xc}))))*(1-c[0])", conservative=False)
            
        reactions = ph.Reactions(2, [reactiona, reactionb])

        # if chem_react_type == "int_maint":
        #     reactiona = ph.Reaction([1, 0], f"-{k_M}*c[0]+{k_U}*c[1]", conservative=False)
        #     reactionb = ph.Reaction([0, 1], f"{k_M}*c[0]-{k_U}*c[1]", conservative=False)
        # elif chem_react_type == "ext_maint":
        #     reactiona = ph.Reaction([1, 0], f"-{k_M}*c[0]+{k_U}*(1-c[0])", conservative=False)
        #     reactionb = ph.Reaction([0, 1], f"{k_M}*c[0]-{k_U}*(1-c[0])", conservative=False)
            
        # reactions = ph.Reactions(2, [reactiona, reactionb])

    else:
        reactions = None

    # set interaction matrix
    chi_m = np.zeros((2, 2))
    chi_m[0, 1] = chi_m[1, 0] = chi
    kappa_m = kappa * np.diag([1.0, 0.0])     # set kappa matrix

    # O: U, 1: M, 2: S

    x, y = GRID.axes_coords
    Y,X = np.meshgrid(x,y)
    f = ph.FloryHugginsNComponents(2, chis=chi_m) #, variables=["a", "b"])

    if advection:      
        velocity_field = np.ones((2, size+2, size+2))
        velocity_field[0,:,:] = u
        velocity_field[1,:,:] = 0
    else:
        velocity_field = None
    
    # define Cahn Hilliard PDE including chemical reactions
    eq = ph.CahnHilliardMultiplePDE(
        {
            "free_energy": f,
            "kappa": kappa_m,
            "mobility": [d, 1], # I think the third value is solvent diffusivity but it is ignored since phi_S = 1 - sum(other phis) so it should not matter?
            "reactions": reactions,
            "velocity_field": velocity_field,
        }
    )

    storage_write = pde.FileStorage(output_filename+".hdf")
    # set trackers
    storage = pde.MemoryStorage()


    trackers = [
        "progress",                                    # show progress bar during simulation
        storage.tracker(interval=interval_for_tracking),            # store data every interval
        storage_write.tracker(interrupts=interval_for_tracking),    
        #pde.PlotTracker(interval="0:10"),             # show images during simulation
        "steady_state",                                # abort when steady state is reached
    ]

    # run simulation
    sol, solver_info = eq.solve(
        STATE,
        simulation_time,
        solver = "explicit", # Euler is default.  Can also be "implicit". 
        #maxerror = 0.001,
        #maxiter = 10,
        dt = 0.0001,           # If adaptive = True, this will only be the initial time step and then it will be adapted
        tolerance=1e-4, #6,
        adaptive=True,
        tracker=trackers,
        ret_info = True
    )

    # save movie

    pde.movie(
        storage,
        f"{output_folder}movie.mp4",
        plot_args={"vmin": 0, "vmax": 1, "cmap":cmapcolor, "subplot_args": [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}]})
    
    # return final state
    solution = {
        "sol0": sol[0].data,
    }

    if save_final_config:
        storage_read = pde.FileStorage(output_filename+".hdf")
        for time, UM_fields in storage_read.items():
            if time == simulation_time:
                    U_field = UM_fields[0]
                    U_field.to_file(output_filename+"_U.hdf5")
                    
    sol.plot(title = "Final Phase Field", colorbar = True, clim = [0,1], cmap = cmapcolor, subplot_args = [{"title": r"$\phi_U$"},{"title": r"$\phi_M$"}])

    return solution, solver_info
