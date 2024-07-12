from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np

# ALl coordinates are given in the cosy centered at the upper edge of the stringer / lower edge of the panel / joint between stringer and panel.


Reuter: Final[np.ndarray] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])


def T(theta_deg: float) -> np.ndarray:  # returns the material cosy to problem cosy transformation matrix
    theta_rad = np.deg2rad(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c], [-s * c, s * c, c ** 2 - s ** 2]])


@dataclass
class PuckFailure:
    RF_FF: float
    RF_IFF: float
    mode: str


class Material:

    def __init__(self, E1_avg: float, E2_avg: float, G12_avg: float, nu12: float, R1t: float, R1c: float, R2t: float,
                 R2c: float, R21: float, p21t: float = 0.25, p21c: float = 0.25, p22t: float = 0.25, p22c: float = 0.25):
        self.E1_avg = E1_avg
        self.E2_avg = E2_avg
        self.G12_avg = G12_avg
        self.nu12 = nu12
        self.nu21 = self.nu12 * self.E2_avg / self.E1_avg
        self.R1t = R1t
        self.R1c = R1c
        self.R2t = R2t
        self.R2c = R2c
        self.R21 = R21
        self.p21t = p21t
        self.p21c = p21c
        self.p22t = p22t
        self.p22c = p22c


@dataclass
class Ply:
    mat: Material
    t: float
    theta: float

    @cached_property
    def Q(self) -> np.ndarray:
        Q11 = self.mat.E1_avg / (1 - self.mat.nu12 * self.mat.nu21)
        Q12 = self.mat.nu12 * self.mat.E2_avg / (1 - self.mat.nu12 * self.mat.nu21)
        Q22 = self.mat.E2_avg / (1 - self.mat.nu12 * self.mat.nu21)
        Q66 = self.mat.G12_avg
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    @cached_property
    def Q_bar(self) -> np.ndarray:
        T_inv = np.linalg.inv(T(self.theta))
        return T_inv @ self.Q @ np.transpose(T_inv)

    def strength_failure(self, sigma_1: float, sigma_2: float, tau_21: float) -> PuckFailure:  # The stress state must be specified in the ply material coordinate system.
        m = self.mat
        RF_FF = (m.R1t if sigma_1 > 0 else m.R1c) / abs(sigma_1)
        R22A = m.R2c / 2 / (1 + m.p22c)
        # tau21c = m.R21 * np.sqrt(1 + 2 * m.p21c * R22A / m.R21)
        tau21c = m.R21 * np.sqrt(1 + 2 * m.p22c)  # according to the Moodle post from 2024-07-09, 17:27
        if sigma_2 >= 0:
            mode = 'A'
            RF_IFF = 1 / (np.sqrt((tau_21 / m.R21) ** 2 + (1 - m.p21t * m.R2t / m.R21) ** 2 * (sigma_2 / m.R2t) ** 2) + m.p21t * sigma_2 / m.R21)
        elif abs(tau_21 / sigma_2) >= abs(tau21c) / R22A:  # It's important to check the inequality this way since tau_21 may be zero.
            mode = 'B'
            RF_IFF = m.R21 / (np.sqrt(tau_21 ** 2 + (m.p21c * sigma_2) ** 2) + m.p21c * sigma_2)
        else:
            mode = 'C'
            RF_IFF = -sigma_2 / m.R2c / ((tau_21 / 2 / (1 + m.p22c) / m.R21) ** 2 + (sigma_2 / m.R2c) ** 2)
        return PuckFailure(RF_FF, RF_IFF, mode)


@dataclass
class Laminate:
    plies: [Ply]
    knockdown_factor: float  # assumed to be equal for all plies in a laminate, i.e., assuming all plies are made from the same materials
    sigma_ult_c: float

    @cached_property
    def t(self) -> float:
        t: float = 0
        for ply in self.plies:
            t += ply.t
        return t

    @cached_property
    def ABD(self) -> np.ndarray:
        A = np.zeros(shape=(3, 3))
        B = np.zeros(shape=(3, 3))
        D = np.zeros(shape=(3, 3))
        z: float = -self.t / 2
        for ply in self.plies:
            A = A + ply.Q_bar * ply.t
            B = B + ply.Q_bar * ((z + ply.t) ** 2 - z ** 2) / 2
            D = D + ply.Q_bar * ((z + ply.t) ** 3 - z ** 3) / 3
            z += ply.t
        # noinspection PyTypeChecker
        return np.block([[A, B], [B, D]])

    def E_hom_avg_x(self, free_lateral_deformation: bool) -> float:
        if free_lateral_deformation:
            # noinspection PyTypeChecker
            return 1 / np.linalg.inv(self.ABD[0:3, 0:3])[0, 0] / self.t
        else:
            # noinspection PyTypeChecker
            return self.ABD[0, 0] / self.t

    def E_hom_B_x(self, free_lateral_deformation: bool) -> float:
        return self.E_hom_avg_x(free_lateral_deformation) * self.knockdown_factor

    def E_hom_avg_x_b(self, free_lateral_deformation: bool, parallel_to_bending_axis: bool) -> float:
        if parallel_to_bending_axis:
            if free_lateral_deformation:
                # noinspection PyTypeChecker
                return 12 / np.linalg.inv(self.ABD[3:6, 3:6])[0, 0] / self.t ** 3
            else:
                # noinspection PyTypeChecker
                return 12 * self.ABD[3, 3] / self.t ** 3
        else:
            return self.E_hom_avg_x(free_lateral_deformation)

    def E_hom_B_x_b(self, free_lateral_deformation: bool, parallel_to_bending_axis: bool) -> float:
        return self.E_hom_avg_x_b(free_lateral_deformation, parallel_to_bending_axis) * self.knockdown_factor

    def G_hom_avg_xy(self, free_lateral_deformation: bool) -> float:
        if free_lateral_deformation:
            # noinspection PyTypeChecker
            return 1 / np.linalg.inv(self.ABD[0:3, 0:3])[2, 2] / self.t
        else:
            # noinspection PyTypeChecker
            return self.ABD[2, 2] / self.t

    def G_hom_B_xy(self, free_lateral_deformation: bool) -> float:
        return self.G_hom_avg_xy(free_lateral_deformation) * self.knockdown_factor


@dataclass
class Panel:
    laminate: Laminate
    length: float
    width: float

    def sigma_x_crit_biaxial(self, sigma_x: float, sigma_y: float, m_max: int, n_max: int) -> float:
        sigma_crit = 999999999
        if sigma_x > 0:
            return sigma_crit
        alpha = self.length / self.width
        beta = sigma_y / sigma_x
        ABD_B = self.laminate.knockdown_factor * self.laminate.ABD
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                sigma_crit_new = np.pi ** 2 / self.width ** 2 / self.laminate.t / ((m / alpha) ** 2 + beta * n ** 2) * (
                        ABD_B[3, 3] * (m / alpha) ** 4 + 2 * (ABD_B[3, 4] + ABD_B[5, 5]) * (m * n / alpha) ** 2 + ABD_B[4, 4] * n ** 4)
                if 0 < sigma_crit_new < sigma_crit:
                    if m == m_max:
                        print('Warning: Reached m_max in Panel.sigma_x_crit_biaxial')
                    if n == n_max:
                        print('Warning: Reached n_max in Panel.sigma_x_crit_biaxial')
                    sigma_crit = sigma_crit_new
        return sigma_crit

    @cached_property
    def tau_xy_crit_shear(self) -> float:
        ABD_B = self.laminate.knockdown_factor * self.laminate.ABD
        t = self.laminate.t
        b = self.width
        delta = np.sqrt(ABD_B[3, 3] * ABD_B[4, 4]) / (ABD_B[3, 4] + 2 * ABD_B[5, 5])
        if delta >= 1:
            tau_crit = 4 / t / b ** 2 * ((ABD_B[3, 3] * ABD_B[4, 4] ** 3) ** 0.25 * (8.12 + 5.05 / delta))
        else:
            tau_crit = 4 / t / b ** 2 * (np.sqrt(ABD_B[4, 4] * (ABD_B[3, 4] + 2 * ABD_B[5, 5])) * (11.7 + 0.532 * delta + 0.938 * delta ** 2))
        return tau_crit

    def RF_panel_buckling(self, sigma_x: float, sigma_y: float, tau_xy: float, m_max: int, n_max: int) -> float:  # m, n: maximum numbers of half waves in local x, y direction used in the computation of sigma_xx_crit_biaxial
        return 1 / (abs(sigma_x) / self.sigma_x_crit_biaxial(sigma_x, sigma_y, m_max, n_max) + (abs(tau_xy) / self.tau_xy_crit_shear) ** 2)

    @cached_property
    def z_EC(self) -> float:
        return -self.laminate.t / 2

    @cached_property
    def area(self) -> float:
        return self.laminate.t * self.width

    @cached_property
    def E_hom_B_x(self) -> float:
        return self.laminate.E_hom_B_x(free_lateral_deformation=False)

    @cached_property
    def E_hom_B_x_b(self) -> float:
        return self.laminate.E_hom_B_x_b(free_lateral_deformation=False, parallel_to_bending_axis=True)

    @cached_property
    def I_yy(self) -> float:
        return self.width * self.laminate.t ** 3 / 12


@dataclass
class Stringer:
    laminate: Laminate
    flange_width: float
    web_height: float

    # Homogenized engineering constants:

    @cached_property
    def E_hom_avg_x(self) -> float:
        return (self.laminate.E_hom_avg_x(free_lateral_deformation=True) * self.web_height + self.laminate.E_hom_avg_x(free_lateral_deformation=False) * self.flange_width) / (self.web_height + self.flange_width)

    @cached_property
    def G_hom_avg_xy(self) -> float:
        return (self.laminate.G_hom_avg_xy(free_lateral_deformation=True) * self.web_height + self.laminate.G_hom_avg_xy(free_lateral_deformation=False) * self.flange_width) / (self.web_height + self.flange_width)

    @cached_property
    def E_hom_B_x_flange(self) -> float:
        return self.laminate.E_hom_B_x(free_lateral_deformation=False)

    @cached_property
    def E_hom_B_x_b_flange(self) -> float:
        return self.laminate.E_hom_B_x_b(free_lateral_deformation=False, parallel_to_bending_axis=True)

    @cached_property
    def E_hom_B_x_web(self) -> float:
        return self.laminate.E_hom_B_x(free_lateral_deformation=True)

    @cached_property
    def E_hom_B_x_b_web(self) -> float:
        return self.laminate.E_hom_B_x_b(free_lateral_deformation=True, parallel_to_bending_axis=False)

    # Crippling analysis:

    @cached_property
    def sigma_crippling_web(self) -> float:
        # sigma_ult_c is not multiplied with the knockdown factor in this course even when it is used for stability analysis (2024-07-05).
        return self.laminate.sigma_ult_c * 1.63 / (self.web_height / self.laminate.t) ** 0.717

    # Strength analysis:

    def ply_strength_failures_from_1D_strain(self, eps_x: float) -> [PuckFailure, ...]:
        strength_failures: list[PuckFailure] = []
        for ply in self.laminate.plies:
            ply_strain_state: np.ndarray = Reuter @ T(ply.theta) @ np.linalg.inv(Reuter) @ np.array([eps_x, 0, 0])
            ply_stress_state: np.ndarray = ply.Q @ ply_strain_state
            strength_failures.append(ply.strength_failure(ply_stress_state[0], ply_stress_state[1], ply_stress_state[2]))
        return strength_failures

    # Stringer geometry:

    @cached_property
    def area_flange(self) -> float:
        return self.laminate.t * self.flange_width

    @cached_property
    def area_web(self) -> float:
        return self.laminate.t * self.web_height

    @cached_property
    def area(self) -> float:
        return self.area_flange + self.area_web

    @cached_property
    def z_EC_flange(self) -> float:
        return self.laminate.t / 2

    @cached_property
    def z_EC_web(self) -> float:
        return self.laminate.t + self.web_height / 2

    @cached_property
    def z_EC(self) -> float:
        return ((self.z_EC_flange * self.laminate.E_hom_avg_x(free_lateral_deformation=False) * self.flange_width + self.z_EC_web * self.laminate.E_hom_avg_x(free_lateral_deformation=True) * self.web_height) /
                (self.flange_width * self.laminate.E_hom_avg_x(free_lateral_deformation=False) + self.web_height * self.laminate.E_hom_avg_x(free_lateral_deformation=True)))

    @cached_property
    def I_yy_flange(self) -> float:
        return self.flange_width * self.laminate.t ** 3 / 12

    @cached_property
    def I_yy_web(self) -> float:
        return self.laminate.t * self.web_height ** 3 / 12

    @cached_property
    def I_yy(self) -> float:
        t = self.laminate.t
        return self.I_yy_flange + self.flange_width * t * (t / 2 - self.z_EC) ** 2 + self.I_yy_web + self.web_height * t * (t + self.web_height / 2 - self.z_EC) ** 2


class StiffenedPanel:  # models a panel stiffened by a single stringer
    def __init__(self, panel: Panel, stringer: Stringer, c: float = 1) -> None:  # assuming low rotational stiffness at the ends of the column
        self.panel = panel
        self.stringer = stringer
        self.c = c

    def average_stress(self, panel_stresses: list[float, ...], stringer_stresses: list[float, ...]) -> float:
        return (self.panel.area * np.average(panel_stresses) + self.stringer.area * np.average(stringer_stresses)) / (self.panel.area + self.stringer.area)

    @cached_property
    def z_EC(self) -> float:
        s = self.stringer
        p = self.panel
        return (((s.z_EC_web * s.laminate.E_hom_avg_x(free_lateral_deformation=True) * s.web_height +
                  s.z_EC_flange * s.laminate.E_hom_avg_x(free_lateral_deformation=False) * s.flange_width) * s.laminate.t +
                 p.z_EC * p.laminate.E_hom_avg_x(free_lateral_deformation=False) * p.width * p.laminate.t) /
                (s.area_web * s.laminate.E_hom_avg_x(free_lateral_deformation=True) +
                 s.area_flange * s.laminate.E_hom_avg_x(free_lateral_deformation=False) +
                 p.area * p.laminate.E_hom_avg_x(free_lateral_deformation=False)))

    @cached_property
    def EI_hom_B_yy(self) -> float:  # used for Euler and Euler-Johnson buckling
        p = self.panel
        s = self.stringer
        return ((p.E_hom_B_x_b * p.I_yy + p.E_hom_B_x * p.area * (p.z_EC - self.z_EC) ** 2 +
                 s.E_hom_B_x_b_flange * s.I_yy_flange + s.E_hom_B_x_flange * s.area_flange * (s.z_EC_flange - self.z_EC) ** 2) +
                s.E_hom_B_x_b_web * s.I_yy_web + s.E_hom_B_x_web * s.area_web * (s.z_EC_web - self.z_EC) ** 2)

    @cached_property
    def I_yy(self) -> float:
        p = self.panel
        s = self.stringer
        return (p.I_yy + p.area * (p.z_EC - self.z_EC) ** 2 +
                s.I_yy_web + s.area_web * (s.z_EC_web - self.z_EC) ** 2 +
                s.I_yy_flange + s.area_flange * (s.z_EC_flange - self.z_EC) ** 2)

    @cached_property
    def E_hom_B_yy(self) -> float:
        return self.EI_hom_B_yy / self.I_yy

    @cached_property
    def area(self) -> float:
        return self.stringer.area + self.panel.area

    @cached_property
    def radius_of_gyration(self) -> float:
        return np.sqrt(self.I_yy / self.area)

    @cached_property
    def slenderness(self) -> float:
        return self.c * self.panel.length / self.radius_of_gyration

    @cached_property
    def critical_slenderness(self) -> float:
        return np.pi * np.sqrt(2 * self.E_hom_B_yy / self.stringer.sigma_crippling_web)

    @cached_property
    def critical_buckling_stress(self) -> float:
        if self.slenderness < self.critical_slenderness:
            return self.stringer.sigma_crippling_web - np.square(self.stringer.sigma_crippling_web * self.slenderness / 2 / np.pi) / self.E_hom_B_yy
        else:
            return np.square(np.pi / self.slenderness) * self.E_hom_B_yy


if __name__ == '__main__':
    pass
