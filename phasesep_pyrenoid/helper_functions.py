from droplets import DiffuseDroplet, Emulsion, SphericalDroplet
import json
import pde
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

scale_length = 14.4 # because we say 7.2 = 1um and dx = 0.5
dt_length = 10/200 # because we want the simulation that lasts 200 to be 10min

def dimensionalize_params(params, params_advec = None, L_char=1.0, T_char=1.0, P_char=1.0):
    """
    Convert dimensionless parameters into dimensional ones using
    characteristic scales:
        - L_char: characteristic length scale
        - T_char: characteristic time scale
        - P_char: characteristic pressure scale
    """

    dim_params = params.copy()

    # Kinetic rates (1/time)
    dim_params["k_M"] = params["k_M"] / T_char
    dim_params["k_U"] = params["k_U"] / T_char

    # Diffusion coeff ~ L^2 / T
    dim_params["d"] = params["d"] * (L_char**2) / T_char

    # Interfacial width ~ L
    dim_params["kappa"] = params["kappa"] * L_char

    # Flory-Huggins χ is dimensionless, keep as is
    dim_params["chi"] = params["chi"]

    # Geometry
    dim_params["shape"] = params["shape"] * L_char
    dim_params["size"] = params["size"]  # number of grid points stays same

    # Simulation time ~ T
    dim_params["simulation_time"] = params["simulation_time"] * T_char

    # Sigmoid params (stay dimensionless unless you want xc ~ concentration scale)
    dim_params["xc"] = params["xc"]
    dim_params["b"] = params["b"]

    if params_advec:
        dim_params_advec = params_advec.copy()

        dim_params_advec["nx"] = params_advec["nx"]
        dim_params_advec["ny"] = params_advec["ny"]
    
        dim_params_advec["Lx"] = params_advec["Lx"] * L_char
        dim_params_advec["Ly"] = params_advec["Ly"] * L_char  # number of grid points stays same    
   

        # Viscosity μ ~ [Pa·min] = [P × T]
        dim_params_advec["mu"] = params_advec["mu"] * P_char * T_char

        # Damping η = [P × T / L^2]
        dim_params_advec["eta"] = params_advec["eta"] * P_char * T_char / L_char**2

        # Pressure P0 ~ P
        dim_params_advec["P0"] = params_advec["P0"] * P_char

        # Pressure width σP ~ P
        dim_params_advec["sigma_P"] = params_advec["sigma_P"] * P_char

    return dim_params, dim_params_advec

def get_initial_radius(parent_folder, ):
    '''Calculate the radius of the drople in equilibrium, to use afterwards to get characteristic lengthscale'''

    storage_read = pde.FileStorage(parent_folder + "final_config.hdf")

    # Read parameters
    with open(parent_folder + "params.json", "r") as f:
        params = json.load(f)

    # Get last timestep directly
    last_idx = params["simulation_time"] - 1
    UM_fields = storage_read[last_idx]
    # Extract the slice at mid-plane
    data = np.transpose(UM_fields.data[0][int(params["size"]/2)])

    plt.figure(figsize = (4,3))
    plt.plot(data)
    plt.xlabel("x axis")
    plt.ylabel("phi_U")

    pyrenoid_init_mask = data>params["xc"]
    idx = np.where(pyrenoid_init_mask)[0]
    dist = idx[-1] - idx[0]
    dx = params["shape"] / (params["size"]-1)
    radius = dist * dx / 2
    return radius

# Define a function to save selected parameters
def save_parameters(params_u, params_m, filename="init_cond_params.json"):
    with open(filename, "w") as f:
        json.dump({"U": params_u, "M": params_m}, f, indent=4)

# Define initial conditions functions
def init_random(grid, rng, params):
    """ Random normal perturbation-based initial state """
    perturbation = pde.ScalarField.random_normal(grid=grid, mean=0, std=params["std"], rng=rng)
    return params["phi"] + perturbation

def init_single_droplet(grid, _, params):
    """ Single droplet initial condition """
    droplet = DiffuseDroplet(position=[params["position"][0], params["position"][1]], radius=params["radius"])
    return params["phi_out"] + params["phi_in"] * droplet.get_phase_field(grid)

def init_emulsion(grid, _, params):
    """ Emulsion with multiple droplets """
    droplets = [DiffuseDroplet(position=[d["position"][0], d["position"][1]], radius=d["radius"]) for d in params["droplets"]]
    emul = Emulsion(droplets)
    return params["phi_out"] + params["phi_in"] * emul.get_phasefield(grid)

def init_from_file(input_filename, params):
    return pde.FieldBase.from_file(input_filename+params["suffix"])


def external_potential(grid, a, xc, c):
    """Defines an external potential on the grid."""
    x, y = grid.axes_coords
    Y,X = np.meshgrid(x,y)
    func = a * np.exp(-(X-xc)**2/(2*c**2))#w*(1-1/(1+np.exp(-b*np.abs((Y-xc))))) ### (a*(X-L/2)**2*4/L**2+0.5)
    return np.array([func, func, func]) 

def double_well_potential(grid, a,b,xc):
    x, y = grid.axes_coords
    Y,X = np.meshgrid(x,y)
    func = 10*(a*((X-xc)/20)**4 - b*((X-xc)/20)**2 + 1/4)
    return  np.array([func, func, func]) 


dark_orange = "#E3812C"

# Define custom colormap
colors = [
    (0.0,  "#FFFBF1"),
    (0.1,  "#FAE4A8"),
    (0.4,  "#F4C777"),
    (0.6,  "#F4B75D"),
    (0.8,  "#E9953A"),
    (1.0,  dark_orange)
]
cmap_orange = LinearSegmentedColormap.from_list("custom_orange", colors)

# Define the custom colormap
colors_exp = [
    (0.0, 0.0, 0.0),   # Black
    (0.2, 0.6, 1.0),   # Blue
    (0.4, 1.0, 0.4),   # Green
    (1.0, 1.0, 0.4),   # Yellow
    (1.0, 0.6, 0.4),   # Orange
    (1.0, 0.2, 0.6),   # Red-Pink
    (0.8, 0.6, 0.8),   # Light Purple
    (1.0, 1.0, 1.0)    # White
]

colors_phases = [
"#C43E96",
"#06948E",
"#825CA6"    
]

# Create a colormap
cmap_exp = LinearSegmentedColormap.from_list("custom_expgradient", colors_exp, N=256)

def make_video(times, dt, UM_fields_list,cmap_exp,filename, fps = 20):# Set up the figure
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')
    im = ax.imshow(np.zeros_like(UM_fields_list[0]),
                cmap=cmap_exp, clim=[0, 1], origin="lower")
    add_scalebar(ax, color = "white")
    title = ax.set_title("")
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\phi_U + \phi_M$")
    # Define update function
    def update(frame):
        time = times[frame]
        UM_fields = UM_fields_list[frame]
        im.set_data(np.transpose(UM_fields))
        time_min = time*dt
        title.set_text(f"t = {time_min:.1f}min")

        return [im, title]

    # Set up writer
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Your Name'), bitrate=1800)

    # Create animation and save
    with writer.saving(fig, filename+".mp4", dpi=200):
        for frame in range(len(times)):
            update(frame)
            writer.grab_frame()

    plt.close()

def add_scalebar(ax, color = "black", scalebar_length = scale_length,  # in µm
        scalebar_x = 8,  # starting x position
        scalebar_y = 12,  # starting y position
):
        # Add a scale bar (1 µm) 
        ax.hlines(scalebar_y, scalebar_x, scalebar_x + scalebar_length, colors=color, linewidth=2)
        ax.text(scalebar_x + scalebar_length / 2, scalebar_y - 2, '1 µm', color=color,
                ha='center', va='top', fontsize=12)

def add_timenote(ax, time,  x = 5, y = 78, color = "black"):
        ax.text(x, y, f't = {time}min', color=color,
                ha='left', va='top', fontsize=10)
        
def get_waistline_profile(storage_read):
    Udata_list1 = []
    for time1, UM_fields in storage_read.items():
        Udata_list1.append(UM_fields.data[0][40,:])
    threshold = 0.8
    dist_list = []
    for n in range(len(Udata_list1)):
        min_d = np.where(Udata_list1[n]>threshold)[0][0] if len(np.where(Udata_list1[n]>threshold)[0])>0 else 0
        max_d = np.where(Udata_list1[n]>threshold)[0][-1] if len(np.where(Udata_list1[n]>threshold)[-1])>0 else 0
        dist_list.append(max_d-min_d)
    return dist_list


def get_waistline_profile_fwhm(storage_read):
    dist_list = []
    times_list = []
    #plt.figure(figsize=(3,2))
    for time, UM_fields in storage_read.items():
        profile = UM_fields.data[0][40, :]  # Take horizontal cross-section at y=40
        half_max = 0.5 #0.5 * np.max(profile)    # Compute half of the maximum height- no I chose static value 0.5

        # Interpolate the profile minus half_max so that we find the zero crossings
        x = np.arange(len(profile))
        f = interp1d(x, profile - half_max, kind='linear', fill_value='extrapolate')
        # if _ == 150:
        #     xnew = np.arange(0,len(profile), 0.5)
        #     plt.plot(x,f(x), 'o')
        #     plt.plot(xnew,f(xnew), '-')
        # Find left and right crossings with half_max
        try:
            left = brentq(f, 0, np.argmax(profile))  # Where the profile rises to half max
            right = brentq(f, np.argmax(profile), len(profile)-1)  # Where it falls to half max
            width = right - left  # FWHM is the distance between those two points
        except ValueError:
            width = 0  # If something goes wrong (e.g. peak too flat), assign width = 0
        times_list.append(time)
        dist_list.append(width)
    return np.array(times_list), np.array(dist_list)

def get_ratio_V1_V2(sel_time, storage_read, threshold = 0.1):
    UM_fields1 = 0
    UM_fields2 = 0

    for time1, UM_fields in storage_read.items():
        if time1 == sel_time:
            UM_fields1 = np.sum(UM_fields.data[0][0:40]>threshold)
            UM_fields2 = np.sum(UM_fields.data[0][40:]>threshold)

    return UM_fields1/UM_fields2  

