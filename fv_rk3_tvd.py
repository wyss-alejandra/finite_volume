import math

import fv_parameters as par
import numpy as np
import matplotlib.pyplot as plt


class FvRk3Tvd:

    def __init__(self, i_order, n_control_volumes, a, b, final_time, co, alpha,
                 discontinuity_point, discontinuity_value_left, discontinuity_value_right, debug=True):

        self.debug = debug

        self.discontinuity_point = discontinuity_point
        self.discontinuity_point_value_left = discontinuity_value_left
        self.discontinuity_point_value_right = discontinuity_value_right

        self.n_control_volumes = n_control_volumes
        self.total_control_volumes = self.n_control_volumes + 6

        self.a = a
        self.b = b
        self.h = (b - a) / self.n_control_volumes

        self.gauss_legendre_value_n = 8  # up to 96
        self.dimensions = 1  # working on 1-D but to keep in mind for a different system
        
        self.i_order = i_order
        
        self.final_time = final_time
        
        self.co = co
        self.alpha = alpha

        # ############################################ Defining variables ##############################################       
        shape_total_control_volumes = (self.total_control_volumes, self.dimensions)
        
        self.x_n = np.zeros(shape_total_control_volumes).reshape(shape_total_control_volumes)
        self.u = np.zeros(shape_total_control_volumes).reshape(shape_total_control_volumes)
        self.u_a = np.zeros(shape_total_control_volumes).reshape(shape_total_control_volumes)
        
        z = np.zeros((self.n_control_volumes + 7, self.dimensions)).reshape(self.n_control_volumes + 7, self.dimensions)
        
        self.u_fig = np.zeros(self.n_control_volumes)
        self.x_fig = np.zeros(self.n_control_volumes)

        # self.u_a_aux = np.empty([self.n_control_volumes, self.dimensions])
        self.u_a_aux = np.empty([self.total_control_volumes, self.dimensions])
        self.u_ex = np.empty([self.total_control_volumes, self.dimensions])
        self.u_ex_fig = np.empty([self.total_control_volumes, self.dimensions])

        # ############################################ x_n initialization ##############################################
        x_n_start = (self.a - self.h / 2) - 2 * self.h
        x_n_stop = -x_n_start + self.b
        self.x_n = np.linspace(start=x_n_start, stop=x_n_stop, num=self.total_control_volumes).reshape(
            shape_total_control_volumes)

        self.x_fig = self.x_n[3:-3]

        # ############################################ z initialization ################################################
        z_start = (self.a - self.h) - 2 * self.h
        z_stop = self.b + 3 * self.h
        self.z = np.linspace(start=z_start, stop=z_stop, num=len(z))

        # ############################################ Physical data #######################################################
        self.k_z, self.v_z, self.q_z = np.zeros(self.n_control_volumes + 7), np.zeros(
            self.n_control_volumes + 7), np.zeros(self.n_control_volumes + 7)

        self.k_z = np.asarray([self.physical_data_k(z_) for z_ in z])
        self.v_z = np.asarray([self.physical_data_v(z_) for z_ in z])
        self.q_z = np.asarray([self.physical_data_q(z_) for z_ in z])

        self.final_solution = np.zeros((n_control_volumes, 2))

    def discontinuity_point_function(self, x):
        if (x >= 0) & (x <= self.discontinuity_point):
            value_ = self.discontinuity_point_value_left
        else:
            value_ = self.discontinuity_point_value_right
        return value_

    def gauss_legendre(self, a=-1, b=1):
        """Function to obtain the Gauss-Legendre quadrature roots and weights
        :param a: range starting point [a, b]
        :param b: range end point [a, b]
        :return:
        x_: roots of the nth legendre polynomial, transformed to range [a, b]
        w_: quadrature weights
        """

        assert(self.gauss_legendre_value_n >= 1)  # Number of sample points and weights must be >= 1
        # Get the roots and quadrature weights
        x, w = np.polynomial.legendre.leggauss(deg=self.gauss_legendre_value_n)

        x_ = np.zeros_like(x)
        w_ = np.zeros_like(w)

        # Transform to [a, b] range
        for i in range(0, self.gauss_legendre_value_n):
            x_[i] = (a + b) / 2 + self.h / 2 * x[i]
            w_[i] = self.h / 2 * w[i]

        return x_, w_

    def gauss_sol(self, z1, z2, return_x_w=False):
        x, w = self.gauss_legendre(a=z1, b=z2)

        sol = 0
        for i in range(0, self.gauss_legendre_value_n):
            sol = sol + w[i] * self.discontinuity_point_function(x[i])

        sol = sol / self.h

        if return_x_w:
            return sol, x, w
        else:
            return sol

    @staticmethod
    def physical_data_k(x, constant_=True):
        if constant_:
            return 0.01e-20
        else:
            return x  # Function of x that is not currently defined

    @staticmethod
    def physical_data_v(x, constant_=True):
        if constant_:
            return 1
        else:
            return 0.5 + abs(math.cos(x)) / 2

    @staticmethod
    def physical_data_q(x, constant_=True):
        if constant_:
            return 1
        else:
            return x  # Function of x that is not currently defined

    @staticmethod
    def u_d_u_func(alpha_, omega_, d, u, d_u, eps, beta):
        alpha_sum, sum_, sum_1 = 0, 0, 0

        for j in range(0, 2):
            alpha_[j] = d[j] / ((eps + beta[j]) ** 2)
            alpha_sum += alpha_[j]
        for j in range(0, 2):
            omega_[j] = alpha_[j] / alpha_sum
        for j in range(0, 2):
            sum_ += omega_[j] * u[j]
            sum_1 += omega_[j] * d_u[j]

        u_output, d_u_output = sum_, sum_1

        return u_output, d_u_output

    def slopes(self, u2, u1):

        return (u2 - u1) / self.h

    @staticmethod
    def alpha_omega_func(alpha_, omega_, d_s, eps, beta):
        alpha_sum = 0
        for j in range(0, 2):
            alpha_[j] = d_s[j] / ((eps + beta[j]) ** 2)
            alpha_sum += alpha_[j]
        for j in range(0, 2):
            omega_[j] = alpha_[j] / alpha_sum

        return alpha_, omega_

    def reconstruction(self, u, x, z):
        u_r_output, u_l_output = np.zeros(self.total_control_volumes), np.zeros(self.total_control_volumes)  # uR, uL
        d_u_r_output, d_u_l_output = np.zeros(self.total_control_volumes), np.zeros(self.total_control_volumes)  # duR, duL

        u_r, u_l = np.zeros(3), np.zeros(3)  # ur, ul
        d_u_r, d_u_l = np.zeros(3), np.zeros(3)  # dur, dul
        d_r, d_l = np.zeros(3), np.zeros(3)  # dR, dL

        alpha_r, alpha_l = np.zeros(3), np.zeros(3)  # alphaR, alphaL
        omega_r, omega_l = np.zeros(3), np.zeros(3)  # omegaR, omegaL

        source_gauss = np.empty([self.total_control_volumes, 2])
        beta = np.empty([3])

        d_s = np.empty([3])
        u_q = np.empty([3])

        alpha_w, omega_w = np.empty([3]), np.empty([3])

        if self.i_order == 0:

            for i1 in range(1, self.total_control_volumes - 1):  # For each control volume boundary

                u2 = u[i1 + 1, 0]
                u1 = u[i1, 0]
                slope = self.slopes(u2, u1)
                u_r_output[i1] = u1 + slope * (z[i1 + 1, 0] - x[i1, 0])
                d_u_r_output[i1] = slope

                source_gauss[i1, 0] = u1
                source_gauss[i1, 1] = u1

        elif self.i_order == 1:  # Linear reconstruction

            for i2 in range(1, self.total_control_volumes - 1):  # For each control volume

                u2 = u[i2, 0]
                u1 = u[i2 - 1, 0]
                slope1 = self.slopes(u2, u1)
                u2 = u[i2 + 1, 0]
                u1 = u[i2, 0]
                slope2 = self.slopes(u2, u1)
                slope = min(slope1, slope2)  # search of minimum slope (slope)

                u2 = u[i2, 0]
                u_l_output[i2] = u2 + slope * (z[i2, 0] - x[i2, 0])
                u_r_output[i2] = u2 + slope * (z[i2 + 1, 0] - x[i2, 0])
                d_u_l_output[i2], d_u_r_output[i2] = slope, slope

                source_gauss[i2, 0] = u2
                source_gauss[i2, 1] = u2

        elif self.i_order == 2:  # WENO - 5

            d_r[0], d_r[1], d_r[2] = 0.3, 0.6, 0.1
            d_l[0], d_l[1], d_l[2] = 0.1, 0.6, 0.3
            eps = 1E-06
            c1 = 13 / 12
            c2 = 1 / 4

            for i3 in range(2, self.total_control_volumes - 2):

                u_i = u[i3, 0]
                u_i_m1 = u[i3 - 1, 0]
                u_i_m2 = u[i3 - 2, 0]

                u_i1 = u[i3 + 1, 0]
                u_i2 = u[i3 + 2, 0]

                u_r[2] = -7 / 6 * u_i_m1 + 11 / 6 * u_i + u_i_m2 / 3  # {i - 2, i - 1, i}
                u_l[2] = 5 / 6 * u_i_m1 + 1 / 3 * u_i - u_i_m2 / 6

                d_u_r[2] = (-3 * u_i_m1 + 2 * u_i + u_i_m2) / self.h  # {i - 2, i - 1, i}
                d_u_l[2] = (u_i - u_i_m1) / self.h

                u_l[1] = u_i_m1 / 3 + 5 / 6 * u_i - u_i1 / 6  # {i - 1, i, i + 1}
                u_r[1] = -u_i_m1 / 6 + 5 / 6 * u_i + u_i1 / 3

                d_u_l[1] = (u_i - u_i_m1) / self.h  # {i - 1, i, i + 1}
                d_u_r[1] = (u_i1 - u_i) / self.h

                u_r[0] = 5 / 6 * u_i1 + u_i / 3 - u_i2 / 6  # {i, i + 1, i + 2}
                u_l[0] = -7 / 6 * u_i1 + 11 * u_i / 6 + u_i2 / 3

                d_u_r[0] = (u_i1 - u_i) / self.h  # {i, i + 1, i + 2}
                d_u_l[0] = (-u_i2 - 2 * u_i + 3 * u_i1) / self.h

                # Piecewise parabolic reconstruction (r=3) - Smoothness indicators - slide 33
                beta[0] = c1 * (u_i - 2 * u_i1 + u_i2) ** 2 + c2 * (3 * u_i - 4 * u_i1 + u_i2) ** 2
                beta[1] = c1 * (u_i_m1 - 2 * u_i + u_i1) ** 2 + c2 * (u_i_m1 - u_i1) ** 2
                beta[2] = c1 * (u_i_m2 - 2 * u_i_m1 + u_i) ** 2 + c2 * (u_i_m2 - 4 * u_i_m1 + 3 * u_i) ** 2

                # Right extrapolated values
                u_r_output[i3], d_u_r_output[i3] = self.u_d_u_func(alpha_r, omega_r, d_r, u_r, d_u_r, eps, beta)
                # Left extrapolated values
                u_l_output[i3], d_u_l_output[i3] = self.u_d_u_func(alpha_l, omega_l, d_l, u_l, d_u_l, eps, beta)

                # WENO FOR REACTIVE TERMS
                # ######################### First Gaussian point #######################################################
                d_s[0] = (210 - math.sqrt(3)) / 1080
                d_s[1] = 11 / 18
                d_s[2] = (210 + math.sqrt(3)) / 1080

                alpha_w, omega_w = self.alpha_omega_func(alpha_w, omega_w, d_s, eps, beta)

                u_q[2] = u_i - (-4 * u_i_m1 + 3 * u_i + u_i_m2) * math.sqrt(3) / 12  # {i - 2, i - 1, i}
                u_q[1] = u_i - (-u_i_m1 + u_i1) * math.sqrt(3) / 12  # {i - 1, i, i + 1}
                u_q[0] = u_i + (u_i2 + 3 * u_i - 4 * u_i1) * math.sqrt(3) / 12  # {i, i + 1, i + 2}

                general_sum = 0
                for j in range(0, 2):
                    general_sum += omega_w[j] * u_q[j]
                source_gauss[i3, 0] = general_sum

                # ######################### Second Gaussian point ######################################################
                d_s[0] = (210 + math.sqrt(3)) / 1080
                d_s[1] = 11 / 18
                d_s[2] = (210 - math.sqrt(3)) / 1080

                alpha_w, omega_w = self.alpha_omega_func(alpha_w, omega_w, d_s, eps, beta)

                u_q[2] = u_i + (-4 * u_i_m1 + 3 * u_i + u_i_m2) * math.sqrt(3) / 12  # {i - 2, i - 1, i}
                u_q[1] = u_i + (-u_i_m1 + u_i1) * math.sqrt(3) / 12  # {i - 1, i, i + 1}
                u_q[0] = u_i - (u_i2 + 3 * u_i - 4 * u_i1) * math.sqrt(3) / 12  # {i, i + 1, i + 2}

                general_sum = 0
                for j in range(0, 2):
                    general_sum += omega_w[j] * u_q[j]
                source_gauss[i3, 1] = general_sum

        output_dict = {
            'u_r': u_r_output,
            'u_l': u_l_output,
            'd_u_r': d_u_r_output,
            'd_u_l': d_u_l_output,
            'source_gauss': source_gauss}

        print("In reconstruction source gauss shape {}".format(source_gauss.shape)) if self.debug else None

        return output_dict

    def op_reconstruction(self, dt, u_r, u_l, d_u_r, d_u_l, source_gauss):
        # Reconstruction of convective fluxes - slide 35
        fc_r, fc_l = np.zeros(self.total_control_volumes), np.zeros(self.total_control_volumes)
        fd_r, fd_l = np.zeros(self.total_control_volumes), np.zeros(self.total_control_volumes)
        fc_lf, fc_lw = np.zeros(self.total_control_volumes), np.zeros(self.total_control_volumes)  # lf = Lax-Friedrichs, lw = Lax-Wendroff

        fc_fo = np.zeros(self.total_control_volumes)
        fd_m = np.zeros(self.total_control_volumes)
        l = np.zeros(self.total_control_volumes)

        f = np.empty([self.total_control_volumes + 1])

        if self.i_order == 0:
            for i in range(2, self.total_control_volumes - 2):
                f[i] = self.k_z[i] * d_u_r[i] - self.v_z[i] * u_r[i]
        else:
            for i in range(2, self.total_control_volumes - 2):
                fc_l[i] = self.v_z[i] * u_l[i]
                fc_r[i] = self.v_z[i + 1] * u_r[i]
                fd_l[i] = self.k_z[i] * d_u_l[i]
                fd_r[i] = self.k_z[i + 1] * d_u_r[i]

        # CONVECTIVE FLUX FORCE
        for i in range(2, self.total_control_volumes - 2):
            fc_lf[i] = (fc_r[i] + fc_l[i + 1]) / 2 + self.h / (2 * dt) * (u_r[i] - u_l[i + 1])
        for i in range(2, self.total_control_volumes - 2):
            u_l_w = (u_l[i + 1] + u_r[i]) / 2 + dt / (2 * self.h) * (fc_r[i] - fc_l[i + 1])
            fc_lw[i] = self.v_z[i + 1] * u_l_w
        for i in range(2, self.total_control_volumes - 2):
            fc_fo[i] = (fc_lw[i] + fc_lf[i]) / 2

        # DIFFUSIVE FLUX Arithmetic mean
        for i in range(2, self.total_control_volumes - 2):
            fd_m[i] = (fd_l[i + 1] + fd_r[i]) / 2

        for i in range(2, self.total_control_volumes - 2):
            f[i] = fd_m[i] - fc_fo[i]

        f[0] = -fc_l[0] + fd_l[0]
        # f[self.total_control_volumes + 1] = -fc_l[self.total_control_volumes + 1] + fd_l[self.total_control_volumes + 1]

        print("f shape {} fc_l shape {} and fd_l shape {}".format(f.shape, fc_l.shape,
                                                                  fd_l.shape)) if self.debug else None

        f[self.total_control_volumes - 1] = -fc_l[self.total_control_volumes - 1] + fd_l[self.total_control_volumes - 1]

        # REACTION Operator:
        for i in range(2, self.total_control_volumes - 2):
            q_u = self.q_z[i] * 1 / 2 * (source_gauss[i, 0] + source_gauss[i, 1])
            l[i] = (f[i] - f[i - 1]) / self.h - q_u

        print("In opreconstruction source gauss shape {}".format(source_gauss.shape)) if self.debug else None

        return l

    def solex(self, alpha, vv, qq, t, x):
        u_ex = math.exp(-qq * t) * ((self.discontinuity_point_value_left + self.discontinuity_point_value_right) / 2 + (
                    self.discontinuity_point_value_right - self.discontinuity_point_value_left) / 2 * math.erf(
            (x - self.discontinuity_point - vv * t) / (2 * math.sqrt(alpha * t))))

        return u_ex

    def initial_concentration_figure(self):
        fig_, ax_ = plt.subplots(1, 1, figsize=(par.PRINT_WIDTH_LARGE, par.PRINT_HEIGHT_SMALL), dpi=300)

        fig_.subplots_adjust(top=0.90)
        fig_.subplots_adjust(bottom=0.50)

        ax_.plot(self.x_fig, self.u_fig, linewidth=1.5, marker='o', markersize=1.5)

        plt.xticks(rotation=45)

        ax_.set_ylabel('Population Ratio')
        ax_.set_xlabel('Date')

        fig_.set_size_inches(par.PRINT_WIDTH, par.PRINT_HEIGHT)

        ax_.title.set_text('Initial Concentration')
        ax_.set_ylabel('Concentration')
        ax_.set_xlabel('x')
        ax_.tick_params(left=False, bottom=False)

        return fig_

    def fv_rk3(self):
        # ############################################ Average of the initial conditions ###############################
        for i1 in range(3, self.n_control_volumes + 3):
            z1 = self.z[i1]
            z2 = self.z[i1 + 1]

            solution = self.gauss_sol(z1=z1, z2=z2)
            self.u_a[i1] = solution

        # Time step
        delta_t = 1000

        for i2 in range(0, self.n_control_volumes):

            if abs(self.v_z[i2]) > 0:
                delta_t_c = self.co * self.h / self.v_z[i2]
                if delta_t_c < delta_t:
                    delta_t = delta_t_c
            else:
                delta_t = 1000000

            if self.k_z[i2] > 0:
                delta_t_d = self.alpha * self.h * self.h / self.k_z[i2]
                if delta_t_d < delta_t:
                    delta_t = delta_t_d

        # Coefficients for Runge - Kutta TVD
        c1 = 1 / 3
        c2 = delta_t / 4
        c3 = 2 * delta_t / 3

        self.u_fig = self.u_a[3:-3]
        self.x_fig = self.x_n[3:-3]

        t = 0
        n_t = math.floor(self.final_time / delta_t)
        print("n_t is {}".format(n_t)) if self.debug else None
        for j in range(0, n_t):

            t += delta_t

            print("t is {}".format(t)) if self.debug else None

            print("u_a shape is {}".format(self.u_a.shape)) if self.debug else None

            # ############################################ RK3-TVD Step 1 ##############################################
            # Neumann homogeneous boundary conditions
            # This is weird, is there another way to assign this???
            self.u_a[2, 0] = self.u_a[3, 0]
            self.u_a[1, 0] = self.u_a[4, 0]
            self.u_a[0, 0] = self.u_a[5, 0]

            print("uas initialization ", self.u_a[2, 0], self.u_a[3, 0], self.u_a[1, 0], self.u_a[4, 0], self.u_a[0, 0], self.u_a[5, 0]) if self.debug else None

            # This is weird, is there another way to assign this??? NOT SURE ABOUT WHAT IS HAPPENING HERE
            self.u_a[self.n_control_volumes + 3, 0] = self.u_a[self.n_control_volumes + 2, 0]
            self.u_a[self.n_control_volumes + 4, 0] = self.u_a[self.n_control_volumes + 1, 0]
            self.u_a[self.n_control_volumes + 5, 0] = self.u_a[self.n_control_volumes + 0, 0]

            recons_output = self.reconstruction(self.u_a, self.x_n, self.z)
            u_r = recons_output['u_r']
            u_l = recons_output['u_l']
            d_u_r = recons_output['d_u_r']
            d_u_l = recons_output['d_u_l']
            source_gauss = recons_output['source_gauss']

            print("in fv_rk3 for j {} source gauss shape {}".format(j, source_gauss.shape)) if self.debug else None
            # print(u_r, u_l, d_u_r, d_u_l) if self.debug else None

            l = self.op_reconstruction(dt=delta_t, u_r=u_r, u_l=u_l, d_u_r=d_u_r, d_u_l=d_u_l, source_gauss=source_gauss)

            for i in range(2, self.n_control_volumes - 2):
                self.u_a_aux[i, 0] = self.u_a[i, 0] + delta_t * l[i]

            # ############################################ RK3-TVD Step 2 ##################################################
            # Neumann homogeneous boundary conditions
            self.u_a_aux[2, 0] = self.u_a_aux[4, 0]
            self.u_a_aux[1, 0] = self.u_a_aux[5, 0]
            self.u_a_aux[0, 0] = self.u_a_aux[6, 0]
            self.u_a_aux[self.n_control_volumes + 3, 0] = self.u_a_aux[self.n_control_volumes + 2, 0]
            self.u_a_aux[self.n_control_volumes + 4, 0] = self.u_a_aux[self.n_control_volumes + 1, 0]
            self.u_a_aux[self.n_control_volumes + 5, 0] = self.u_a_aux[self.n_control_volumes, 0]

            recons_output = self.reconstruction(self.u_a_aux, self.x_n, self.z)
            u_r = recons_output['u_r']
            u_l = recons_output['u_l']
            d_u_r = recons_output['d_u_r']
            d_u_l = recons_output['d_u_l']
            source_gauss = recons_output['source_gauss']

            l = self.op_reconstruction(dt=delta_t, u_r=u_r, u_l=u_l, d_u_r=d_u_r, d_u_l=d_u_l, source_gauss=source_gauss)

            for i in range(2, self.total_control_volumes - 2):
                self.u_a_aux[i, 0] = 0.75 * self.u_a[i] + 0.25 * self.u_a_aux[i, 0] + c2 * l[i]

            # ############################################ RK3-TVD Step 3 ##################################################
            # Neumann homogeneous boundary conditions
            self.u_a_aux[2, 0] = self.u_a_aux[3, 0]
            self.u_a_aux[1, 0] = self.u_a_aux[4, 0]
            self.u_a_aux[0, 0] = self.u_a_aux[5, 0]
            self.u_a_aux[self.n_control_volumes + 3, 0] = self.u_a_aux[self.n_control_volumes + 3, 0]
            self.u_a_aux[self.n_control_volumes + 4, 0] = self.u_a_aux[self.n_control_volumes + 2, 0]
            self.u_a_aux[self.n_control_volumes + 5, 0] = self.u_a_aux[self.n_control_volumes + 1, 0]

            recons_output = self.reconstruction(self.u_a_aux, self.x_n, self.z)
            u_r = recons_output['u_r']
            u_l = recons_output['u_l']
            d_u_r = recons_output['d_u_r']
            d_u_l = recons_output['d_u_l']
            source_gauss = recons_output['source_gauss']

            l = self.op_reconstruction(dt=delta_t, u_r=u_r, u_l=u_l, d_u_r=d_u_r, d_u_l=d_u_l, source_gauss=source_gauss)

            for ii in range(2, self.total_control_volumes - 2):
                self.u[ii, 0] = c1 * (self.u_a[ii, 0] + 2 * self.u_a_aux[ii, 0]) + c3 * l[ii]

            for ii in range(3, self.total_control_volumes - 2):
                self.u_a[ii, 0] = self.u[ii, 0]
                self.u_ex[ii, 0] = self.solex(self.k_z[ii], self.v_z[ii], self.q_z[ii], t, self.x_n[ii])

            jj = 0
            for ii in range(3, self.n_control_volumes + 3):
                self.u_fig[jj, 0] = self.u_a[ii, 0]
                self.u_ex_fig[jj, 0] = self.u_ex[ii, 0]
                jj += 1

            jj = 0
            for ii in range(3, self.n_control_volumes + 3):
                self.x_fig[jj, 0] = self.x_n[ii, 0]
                jj += 1

        for ii in range(0, self.n_control_volumes):
            self.final_solution[ii, 0] = self.x_fig[ii]
            self.final_solution[ii, 1] = self.u_fig[ii]
