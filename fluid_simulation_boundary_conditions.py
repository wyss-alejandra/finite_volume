# https://philip-mocz.medium.com/create-your-own-finite-volume-fluid-simulation-with-python-part-2-boundary-conditions-source-bda6994b4645
# Rayleigh-Taylor Instability
# Occurs when a heavy fluid sits on top of a light fluid and is pulled down by gravity.
# Setting boundary conditions by adding ghost cells.


# Other types of boundary conditions are possible and may be implemented using ghost cells.
# A common one is Dirichlet boundary conditions, where one gives the ghost cells a prescribed value.

import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Finite Volume Fluid Simulation (With Python) Part 2: 
Boundary Conditions and Source Terms
Philip Mocz (2021), @PMocz
Simulate the Raleigh-Taylor Instability with the Finite Volume Method. 
Demonstrates gravity source term and Reflecting boundary condition
"""


def get_conserved(rho, vx, vy, p, gamma, vol):
    """ Calculate the conserved variable from the primitive

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
    """
    Calculate the primitive variable from the conservative
    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Energy   is matrix of energy in cells
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

    # Note that when calculating the primitive variables from the conservative state, the ghost cell values are set
    rho, vx, vy, p = set_ghost_cells(rho, vx, vy, p)

    return rho, vx, vy, p


def get_gradient(f, dx):
    """
    Calculate the gradients of a field
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

    # Note that we are calculating the gradients, the ghost cell primitive variable gradients are also being set
    f_dx, f_dy = set_ghost_gradients(f_dx, f_dy)

    return f_dx, f_dy


def slope_limit(f, dx, f_dx, f_dy):
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
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    dx       is the cell size
    f_xl     is a matrix of spatial-extrapolated values on `left' face along x-axis
    f_xr     is a matrix of spatial-extrapolated values on `right' face along x-axis
    f_yr     is a matrix of spatial-extrapolated values on `left' face along y-axis
    f_yr     is a matrix of spatial-extrapolated values on `right' face along y-axis
    """
    # directions for np.roll()
    r = -1  # right
    l = 1  # left

    f_xl = f - f_dx * dx / 2
    f_xl = np.roll(f_xl, r, axis=0)
    f_xr = f + f_dx * dx / 2

    f_yl = f - f_dy * dx / 2
    f_yl = np.roll(f_yl, r, axis=1)
    f_yr = f + f_dy * dx / 2

    return f_xl, f_xr, f_yl, f_yr


def apply_fluxes(f, flux_f_x, flux_f_y, dx, dt):
    """
    Apply fluxes to conserved variables
    F        is a matrix of the conserved variable field
    flux_F_X is a matrix of the x-dir fluxes
    flux_F_Y is a matrix of the y-dir fluxes
    dx       is the cell size
    dt       is the time-step
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


def get_flux(rho_l, rho_r, vx_L, vx_r, vy_L, vy_r, p_l, p_r, gamma):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    P_L          is a matrix of left-state  pressure
    P_R          is a matrix of right-state pressure
    gamma        is the ideal gas gamma
    flux_mass    is the matrix of mass fluxes
    flux_momx    is the matrix of x-momentum fluxes
    flux_momy    is the matrix of y-momentum fluxes
    flux_energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_l = p_l / (gamma - 1) + 0.5 * rho_l * (vx_L ** 2 + vy_L ** 2)
    en_r = p_r / (gamma - 1) + 0.5 * rho_r * (vx_r ** 2 + vy_r ** 2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_l + rho_r)
    momx_star = 0.5 * (rho_l * vx_L + rho_r * vx_r)
    momy_star = 0.5 * (rho_l * vy_L + rho_r * vy_r)
    en_star = 0.5 * (en_l + en_r)

    p_star = (gamma - 1) * (en_star - 0.5 * (momx_star ** 2 + momy_star ** 2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_mass = momx_star
    flux_momx = momx_star ** 2 / rho_star + p_star
    flux_momy = momx_star * momy_star / rho_star
    flux_energy = (en_star + p_star) * momx_star / rho_star

    # find wavespeeds
    c_l = np.sqrt(gamma * p_l / rho_l) + np.abs(vx_L)
    c_r = np.sqrt(gamma * p_r / rho_r) + np.abs(vx_r)
    c = np.maximum(c_l, c_r)

    # add stabilizing diffusive term
    flux_mass -= c * 0.5 * (rho_l - rho_r)
    flux_momx -= c * 0.5 * (rho_l * vx_L - rho_r * vx_r)
    flux_momy -= c * 0.5 * (rho_l * vy_L - rho_r * vy_r)
    flux_energy -= c * 0.5 * (en_l - en_r)

    return flux_mass, flux_momx, flux_momy, flux_energy


def add_ghost_cells(rho, vx, vy, p):
    """
    Add ghost cells to the top and bottom

    Ghost cells are added due to the boundary conditions.
    Add a top and bottom layer of ghost cells to the domain.
    This function adds a row to the top and bottom of the primitive variable simulation matrices (i.e., the density, velocity, and pressure)
    prior to the time integration loop.

    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    P        is matrix of cell pressures
    """
    rho = np.hstack((rho[:, 0:1], rho, rho[:, -1:]))
    vx = np.hstack((vx[:, 0:1], vx, vx[:, -1:]))
    vy = np.hstack((vy[:, 0:1], vy, vy[:, -1:]))
    p = np.hstack((p[:, 0:1], p, p[:, -1:]))

    return rho, vx, vy, p


def set_ghost_cells(rho, vx, vy, P):
    """
    Set ghost cells at the top and bottom

    The ghost cells are set to be mirror reflections of the interior neighbors.
    Mirror reflections means copy the value and changes the sign of the normal component (y-component) of the velocity.

    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    P        is matrix of cell pressures
    """

    rho[:, 0] = rho[:, 1]
    vx[:, 0] = vx[:, 1]
    vy[:, 0] = -vy[:, 1]
    P[:, 0] = P[:, 1]

    rho[:, -1] = rho[:, -2]
    vx[:, -1] = vx[:, -2]
    vy[:, -1] = -vy[:, -2]
    P[:, -1] = P[:, -2]

    return rho, vx, vy, P


def set_ghost_gradients(f_dx, f_dy):
    """
    Set ghost cell y-gradients at the top and bottom to be reflections

    This functions sets the gradients to be reflective, by negating their value.

    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    """

    f_dy[:, 0] = -f_dy[:, 1]  # [:, 0] is a ghost cell, and it is assigned to the neighbor value with a (-) change of sign -> negating their value
    f_dy[:, -1] = -f_dy[:, -2]

    return f_dx, f_dy


def add_source_term(mass, momx, momy, energy, g, dt):
    """
    Add gravitational source term to conservative variables

    Updates the conservative variables Q with the sources given at a time-step delta_t
    Source terms account for the change in momentum and energy due to the gravitational field
    Note that this function updates only energy and momentum due to the above sentence

    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Energy   is matrix of energy in cells
    g        is strength of gravity
    Y        is matrix of y positions of cells
    dt       is time-step to progress solution
    """

    energy += dt * momy * g  # note that the energy is updated before the momentum, in order to use the initial value of the momentum (because for momentum we need to use this new value)
    momy += dt * mass * g

    return mass, momx, momy, energy


def main():
    """ Finite Volume simulation """

    # Uses a second-order approach to be consistent with the original time integration -> kick-drift-kick
    # Add the source term contribution over half a time-step (delta_t/2), a "kick"
    # Update the primitive variables
    # Compute and add the fluid fluxes, a "drift"
    # Complete the time-step with a final half-step "kick"

    # Simulation parameters
    # 2D domain [0, 0.5] x [0, 1.5]
    n = 64  # resolution N x 3N
    boxsize_x = 0.5  # 2D domain [0, 0.5]
    boxsize_y = 1.5  # 2D domain [0, 1.5]
    gamma = 1.4  # ideal gas gamma
    courant_fac = 0.4
    t = 0
    t_end = 15
    t_out = 0.1  # draw frequency
    use_slope_limiting = False
    plot_real_time = True  # switch on for plotting as the simulation goes along

    # Mesh
    dx = boxsize_x / n
    vol = dx ** 2
    xlin = np.linspace(0.5 * dx, boxsize_x - 0.5 * dx, n)
    ylin = np.linspace(0.5 * dx, boxsize_y - 0.5 * dx, 3 * n)
    y, x = np.meshgrid(ylin, xlin)

    # Generate Initial Conditions - heavy fluid on top of light, with perturbation
    # This is a Rayleigh-Taylor Instability simulation
    g = -0.1  # gravity
    w0 = 0.0025
    p0 = 2.5
    rho = 1. + (y > 0.75)
    vx = np.zeros(x.shape)  # The initial x-velocity is 0
    vy = w0 * (1 - np.cos(4 * np.pi * x)) * (1 - np.cos(4 * np.pi * y / 3))  # The y-velocity has a single-mode perturbation
    p = p0 + g * (y - 0.75) * rho  # Initial pressure

    # After generating the initial conditions, we also add the ghost cells
    rho, vx, vy, p = add_ghost_cells(rho, vx, vy, p)

    # Get conserved variables
    mass, momx, momy, energy = get_conserved(rho, vx, vy, p, gamma, vol)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    output_count = 1

    # Simulation Main Loop
    while t < t_end:

        # get Primitive variables
        rho, vx, vy, p = get_primitive(mass, momx, momy, energy, gamma, vol)

        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (np.sqrt(gamma * p / rho) + np.sqrt(vx ** 2 + vy ** 2)))
        plot_this_turn = False
        if t + dt > output_count * t_out:
            dt = output_count * t_out - t
            plot_this_turn = True

        # Add Source (half-step)
        # Add a half-step "kick" source term and update the primitive variables before the flux calculation
        mass, momx, momy, energy = add_source_term(mass, momx, momy, energy, g, dt / 2)

        # get Primitive variables
        rho, vx, vy, p = get_primitive(mass, momx, momy, energy, gamma, vol)

        # calculate gradients
        rho_dx, rho_dy = get_gradient(rho, dx)
        vx_dx, vx_dy = get_gradient(vx, dx)
        vy_dx, vy_dy = get_gradient(vy, dx)
        p_dx, p_dy = get_gradient(p, dx)

        # slope limit gradients
        if use_slope_limiting:
            rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)
            p_dx, p_dy = slope_limit(p, dx, p_dx, p_dy)

        # extrapolate half-step in time
        rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * p_dx)
        vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * p_dy)
        p_prime = p - 0.5 * dt * (gamma * p * (vx_dx + vy_dy) + vx * p_dx + vy * p_dy)

        # extrapolate in space to face centers
        rho_xl, rho_xr, rho_yl, rho_yr = extrapolate_in_space_to_face(rho_prime, rho_dx, rho_dy, dx)
        vx_xl, vx_xr, vx_yl, vx_yr = extrapolate_in_space_to_face(vx_prime, vx_dx, vx_dy, dx)
        vy_xl, vy_xr, vy_yl, vy_yr = extrapolate_in_space_to_face(vy_prime, vy_dx, vy_dy, dx)
        p_xl, p_xr, p_yl, p_yr = extrapolate_in_space_to_face(p_prime, p_dx, p_dy, dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_mass_x, flux_momx_x, flux_momy_x, flux_energy_x = get_flux(rho_xl, rho_xr, vx_xl, vx_xr, vy_xl, vy_xr, p_xl,
                                                                        p_xr, gamma)
        flux_mass_y, flux_momy_y, flux_momx_y, flux_energy_y = get_flux(rho_yl, rho_yr, vy_yl, vy_yr, vx_yl, vx_yr, p_yl,
                                                                        p_yr, gamma)

        # update solution
        mass = apply_fluxes(mass, flux_mass_x, flux_mass_y, dx, dt)
        momx = apply_fluxes(momx, flux_momx_x, flux_momx_y, dx, dt)
        momy = apply_fluxes(momy, flux_momy_x, flux_momy_y, dx, dt)
        energy = apply_fluxes(energy, flux_energy_x, flux_energy_y, dx, dt)

        # Add Source (half-step)
        # Add another half-step "kick" after the flux calculation
        mass, momx, momy, energy = add_source_term(mass, momx, momy, energy, g, dt / 2)

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
    plt.savefig('finite_volume2.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
