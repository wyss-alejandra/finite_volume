# https://levelup.gitconnected.com/create-your-own-finite-volume-fluid-simulation-with-python-8f9eab0b8305
# Kelvin-Helmholtz Instability
# Periodic boundary conditions.

# ######## Primitive variables ########
# Density ρ
# Velocity vₓ , vᵧ
# Pressure P

# ######## Conservative variables ########
# Mass Density ρ
# Momentum Density ρvₓ , ρvᵧ
# Energy Density ρe

# The energy density relates to the pressure through the fluid's equation of state
# P = (gamma - 1)rho*u
# where u is the internal energy (that is, temperature) of the fluid, which relates to the total energy e as
# e = u + (vx**2 + xy**2)/2

# Gamma is the ideal gas adiabatic index parameter, for example a monatomic ideal gas ha a gamma=5/3


import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz (2020) Princeton University, @PMocz
Simulate the Kelvin Helmholtz Instability
In the compressible Euler equations
"""


def get_conserved(rho, vx, vy, p, gamma, vol):
    """Calculate the conserved variable from the primitive

    This function translates primitive variables to conserved quantities Q

    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    P        is matrix of cell pressures
    gamma    is ideal gas gamma
    vol      is cell volume
    mass     is matrix of mass in cells
    momx     is matrix of x-momentum in cells
    momy     is matrix of y-momentum in cells
    energy   is matrix of energy in cells
    """
    mass = rho * vol
    momx = rho * vx * vol
    momy = rho * vy * vol
    energy = (p / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)) * vol

    return mass, momx, momy, energy


def get_primitive(mass, momx, momy, energy, gamma, vol):
    """Calculate the primitive variable from the conservative

    This function translates conserved quantities Q into primitive variables

    mass     is matrix of mass in cells
    momx     is matrix of x-momentum in cells
    momy     is matrix of y-momentum in cells
    energy   is matrix of energy in cells
    gamma    is ideal gas gamma
    vol      is cell volume
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    p        is matrix of cell pressures
    """
    rho = mass / vol
    vx = momx / rho / vol
    vy = momy / rho / vol
    p = (energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)

    return rho, vx, vy, p


def get_gradient(f, dx):
    """Calculate the gradients of a field

    Calculates the gradient on a periodic domain in a vectorized fashion (that is, acting on a matrix of variables all at once)

    The second-order finite difference formula
    { (f_{i+1,j}-f{i-1,j})/(2*deltax) , (f_{i,j+1}-f{i,j-1})/(2*deltax)}

    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    """
    # directions for np.roll()
    r = -1  # right
    l = 1  # left

    f_dx = (np.roll(f, r, axis=0) - np.roll(f, l, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, r, axis=1) - np.roll(f, l, axis=1)) / (2 * dx)

    return f_dx, f_dy


def slop_limit(f, dx, f_dx, f_dy):
    """
    Apply slope limiter to slopes
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    """
    # directions for np.roll()
    r = -1  # right
    l = 1  # left

    f_dx = np.maximum(0., np.minimum(1., ((f - np.roll(f, l, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f - np.roll(f, r, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dy = np.maximum(0., np.minimum(1., ((f - np.roll(f, l, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy
    f_dy = np.maximum(0., np.minimum(1., (-(f - np.roll(f, r, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy

    return f_dx, f_dy


def extrapolate_in_space_to_face(f, f_dx, f_dy, dx):
    """
    Calculate the gradients of a field

    Second-Order Extrapolation in Space
    Use the estimated gradients to extrapolate values along a distance delta_x/2 from the cell center to a face
    For example:
    f_{i+1/2,1}~=f_{i,j} + d_d f_{i,j} x delta_x /2

    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    dx       is the cell size
    f_xl     is a matrix of spatial-extrapolated values on `left' face along x-axis
    f_xr     is a matrix of spatial-extrapolated values on `right' face along x-axis
    f_yl     is a matrix of spatial-extrapolated values on `left' face along y-axis
    f_yr     is a matrix of spatial-extrapolated values on `right' face along y-axis
    """
    # directions for np.roll()
    r = -1  # right
    l = 1  # left

    f_xl = f - f_dx * dx / 2  # spatially extrapolating from a cell i,j to the face (i-1/2,j)
    f_xl = np.roll(f_xl, r, axis=0)
    f_xr = f + f_dx * dx / 2  # spatially extrapolating from a cell i,j to the face (i+1/2,j)

    f_yl = f - f_dy * dx / 2
    f_yl = np.roll(f_yl, r, axis=1)
    f_yr = f + f_dy * dx / 2

    return f_xl, f_xr, f_yl, f_yr


def apply_fluxes(f, flux_f_x, flux_f_y, dx, dt):
    """
    Apply fluxes to conserved variables

    Once the fluxes are computed, they can be applied to the conserved fluid quantities Q in each cell

    f        is a matrix of the conserved variable field
    flux_f_x is a matrix of the x-dir fluxes
    flux_f_y is a matrix of the y-dir fluxes
    dx       is the cell size
    dt       is the time step
    """
    # directions for np.roll()
    r = -1  # right
    l = 1  # left

    # update solution
    f += - dt * dx * flux_f_x
    f += dt * dx * np.roll(flux_f_x, l, axis=0)
    f += - dt * dx * flux_f_y
    f += dt * dx * np.roll(flux_f_y, l, axis=1)

    return f


def get_flux(rho_l, rho_r, vx_l, vx_r, vy_l, vy_r, p_l, p_r, gamma):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule

    A simple robust approximation is the Rusanov flux:
    f = 1/2 * (f_l + f_r) - c_max/2 * (u_r - u_l)
    The first term is a simple average of the fluxes as derived from the left or the right fluid variables.
    Then, there is an added term which creates numerical diffusivity. It keeps the solution stable.
    c_max is the maximum signal speed

    rho_l        is a matrix of left-state  density
    rho_r        is a matrix of right-state density
    vx_l         is a matrix of left-state  x-velocity
    vx_r         is a matrix of right-state x-velocity
    vy_l         is a matrix of left-state  y-velocity
    vy_r         is a matrix of right-state y-velocity
    p_l          is a matrix of left-state  pressure
    p_r          is a matrix of right-state pressure
    gamma        is the ideal gas gamma
    flux_mass    is the matrix of mass fluxes
    flux_momx    is the matrix of x-momentum fluxes
    flux_momy    is the matrix of y-momentum fluxes
    flux_energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_l = p_l / (gamma - 1) + 0.5 * rho_l * (vx_l ** 2 + vy_l ** 2)
    en_r = p_r / (gamma - 1) + 0.5 * rho_r * (vx_r ** 2 + vy_r ** 2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_l + rho_r)
    momx_star = 0.5 * (rho_l * vx_l + rho_r * vx_r)  # momentum is rho * v -> this is an average of momentum_x l and r
    momy_star = 0.5 * (rho_l * vy_l + rho_r * vy_r)  # this is an average of momentum_y l and r
    en_star = 0.5 * (en_l + en_r)

    # This comes from p = (gamma - 1) * rho * u where e=u+(v_x**2+v_y**2)/2
    # BUT I don't understand why it's divided by rho_star, because I think it should be multiplying
    p_star = (gamma - 1) * (en_star - 0.5 * (momx_star ** 2 + momy_star ** 2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    # Remember: f = 1/2 * (f_l + f_r) - c_max/2 * (u_r - u_l)
    # It's being calculated in two steps, first without the c_max and then adding the part with the c_max
    flux_mass = momx_star
    flux_momx = momx_star ** 2 / rho_star + p_star
    flux_momy = momx_star * momy_star / rho_star
    flux_energy = (en_star + p_star) * momx_star / rho_star

    # find wavespeeds
    # remember c = (gamma * P / rho) ^ (1/2)
    # Here we are calculating the c left and right, and then the max
    c_l = np.sqrt(gamma * p_l / rho_l) + np.abs(vx_l)  # WHY are we adding this absolute value of vx_ ?
    c_r = np.sqrt(gamma * p_r / rho_r) + np.abs(vx_r)
    c = np.maximum(c_l, c_r)

    # add stabilizing diffusive term
    # Remember: f = 1/2 * (f_l + f_r) - c_max/2 * (u_r - u_l)
    # Here we are adding the c_max part
    flux_mass -= c * 0.5 * (rho_l - rho_r)
    flux_momx -= c * 0.5 * (rho_l * vx_l - rho_r * vx_r)
    flux_momy -= c * 0.5 * (rho_l * vy_l - rho_r * vy_r)
    flux_energy -= c * 0.5 * (en_l - en_r)

    return flux_mass, flux_momx, flux_momy, flux_energy


def main():
    """ Finite Volume simulation """

    # Simulation parameters
    n = 128  # resolution
    boxsize = 1.
    gamma = 5 / 3  # ideal gas gamma
    courant_fac = 0.4
    t = 0
    t_end = 2
    t_out = 0.02  # draw frequency
    use_slope_limiting = False
    plot_real_time = True  # switch on for plotting as the simulation goes along

    # Mesh
    dx = boxsize / n
    vol = dx ** 2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, n)
    y, x = np.meshgrid(xlin, xlin)

    # Generate Initial Conditions - opposite moving streams with perturbation
    # Specifying the initial primitive variables (density, velocity, pressure fields), and the ideal gas parameter
    w0 = 0.1
    sigma = 0.05 / np.sqrt(2.)
    rho = 1. + (np.abs(y - 0.5) < 0.25)
    vx = -0.5 + (np.abs(y - 0.5) < 0.25)
    vy = w0 * np.sin(4 * np.pi * x) * (
                np.exp(-(y - 0.25) ** 2 / (2 * sigma ** 2)) + np.exp(-(y - 0.75) ** 2 / (2 * sigma ** 2)))
    p = 2.5 * np.ones(x.shape)

    # Get conserved variables
    mass, momx, momy, energy = get_conserved(rho, vx, vy, p, gamma, vol)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    output_count = 1

    # Simulation Main Loop
    # Time integration: at each time-step the Finite Volume method will:
    # - Get cell-centered primitive variables from conservative variables -> get_primitive
    # - Calculate the next time-step delta_t -> CFL Courant-Friedrichs-Lewy
    # - Extrapolate primitive variables in time by delta_t/2 using gradients
    # - Feed in face left and right fluid states to compute the fluxes across each face
    # - Update the solution by applying fluxes to the conservative variables

    while t < t_end:

        # get Primitive variables
        rho, vx, vy, p = get_primitive(mass, momx, momy, energy, gamma, vol)

        # get time step (CFL) = dx / max signal speed
        # Conceptually, what the CFL condition says is that in the duration of a time step, the max signal speed may not travel more than the length of a cell
        dt = courant_fac * np.min(dx / (np.sqrt(gamma * p / rho) + np.sqrt(vx ** 2 + vy ** 2)))  # this is the CFL condition
        plot_this_turn = False
        if t + dt > output_count * t_out:
            dt = output_count * t_out - t
            plot_this_turn = True

        # calculate gradients
        rho_dx, rho_dy = get_gradient(rho, dx)
        vx_dx, vx_dy = get_gradient(vx, dx)
        vy_dx, vy_dy = get_gradient(vy, dx)
        p_dx, p_dy = get_gradient(p, dx)

        # slope limit gradients
        if use_slope_limiting:
            rho_dx, rho_dy = slop_limit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slop_limit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slop_limit(vy, dx, vy_dx, vy_dy)
            p_dx, p_dy = slop_limit(p, dx, p_dx, p_dy)

        # extrapolate half-step in time
        # To make the method more accurate it is useful to extrapolate in time by half a time-step before calculating the fluxes
        # This is done by expressing the time gradient simply as a function of spatial gradients (which we know)
        # using the primitive form of the Euler equations
        rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * p_dx)
        vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * p_dy)
        p_prime = p - 0.5 * dt * (gamma * p * (vx_dx + vy_dy) + vx * p_dx + vy * p_dy)

        # extrapolate in space to face centers
        # Second-Order Extrapolation in Space
        # Use estimated gradients to extrapolate values along a distance delta_x / 2 from the cell center to a face
        # It turns out that in general it is better to extrapolate primitive variables and convert back to conservative,
        # rather than extrapolate conservative variables directly,
        # in order to ensure the pressure does not accidentally get reconstructed to negative values due to truncation errors
        rho_xl, rho_xr, rho_yl, rho_yr = extrapolate_in_space_to_face(rho_prime, rho_dx, rho_dy, dx)
        vx_xl, vx_xr, vx_yl, vx_yr = extrapolate_in_space_to_face(vx_prime, vx_dx, vx_dy, dx)
        vy_xl, vy_xr, vy_yl, vy_yr = extrapolate_in_space_to_face(vy_prime, vy_dx, vy_dy, dx)
        p_xl, p_xr, p_yl, p_yr = extrapolate_in_space_to_face(p_prime, p_dx, p_dy, dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        # The heart of the Finite Volume lies in calculating the numerical flux, given a fluid state u (expressed as a collection of conservative variables)
        # on the left and right sides of the interface
        # This may be done in a number of ways, with different levels of accuracy
        flux_mass_x, flux_momx_x, flux_momy_x, flux_energy_x = get_flux(rho_xl, rho_xr, vx_xl, vx_xr, vy_xl, vy_xr, p_xl,
                                                                        p_xr, gamma)
        flux_mass_y, flux_momy_y, flux_momx_y, flux_energy_y = get_flux(rho_yl, rho_yr, vy_yl, vy_yr, vx_yl, vx_yr, p_yl,
                                                                        p_yr, gamma)

        # update solution
        mass = apply_fluxes(mass, flux_mass_x, flux_mass_y, dx, dt)
        momx = apply_fluxes(momx, flux_momx_x, flux_momx_y, dx, dt)
        momy = apply_fluxes(momy, flux_momy_x, flux_momy_y, dx, dt)
        energy = apply_fluxes(energy, flux_energy_x, flux_energy_y, dx, dt)

        # update time
        t += dt

        # plot in real time - color 1/2 particles blue, other half red
        if (plot_real_time and plot_this_turn) or (t >= t_end):
            plt.cla()
            plt.imshow(rho.T)
            plt.clim(0.8, 2.2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)
            output_count += 1

    # Save figure
    plt.savefig('finite_volume.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
