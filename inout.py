import configparser as cp
import csv

import pandas as pd

from composites import *

'''
Possible second arguments of csv.open():
"r"     Read-only, starts at beginning of file (default mode).
"r+"    Read-write, starts at beginning of file.
"w"     Write-only, truncates existing file to zero length or creates a new file for writing.
"w+"    Read-write, truncates existing file to zero length or creates a new file for reading and writing.
"a"     Write-only, starts at end of file if file exists, otherwise creates a new file for writing.
"a+"    Read-write, starts at end of file if file exists, otherwise creates a new file for reading and writing.
"b"     Binary file mode (may appear with any of the key letters listed above). 
        Suppresses EOL <-> CRLF conversion on Windows. 
        And sets external encoding to ASCII-8BIT unless explicitly specified.
"t"     Text file mode (may appear with any of the key letters listed above except "b").
'''


def prepare_csv_for_pandas(filename, delimiter=',') -> None:  # adds commas or semicolons such that every row has the same number of cells
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        rows = list(reader)
    max_delimiters = max(len(row) - 1 for row in rows)
    for row in rows:
        missing_delimiters = max_delimiters - (len(row) - 1)
        row.extend([''] * missing_delimiters)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(rows)


def read_config() -> cp.ConfigParser():
    config = cp.ConfigParser()
    config.read('config.ini')
    return config


def read_ASE_Project2024_task2_1_Template_xxxxxxx(config: cp.ConfigParser) -> pd.DataFrame:
    template_path = config['files']['template_path']
    prepare_csv_for_pandas(filename=template_path, delimiter=';')
    # noinspection PyTypeChecker
    return pd.read_csv(template_path, delimiter=';', na_values=None, keep_default_na=False, header=None)


def create_stiffened_panel_from_config(config: cp.ConfigParser) -> StiffenedPanel:
    template = read_ASE_Project2024_task2_1_Template_xxxxxxx(config=config)

    # Create stiffened panel object
    material = Material(E1_avg=float(template.iat[6, 1]), E2_avg=float(template.iat[7, 1]), G12_avg=float(template.iat[8, 1]), nu12=float(config['params']['nu12']), R1t=float(config['params']['R1t']), R1c=float(config['params']['R1c']),
                        R2t=float(config['params']['R2t']), R2c=float(config['params']['R2c']), R21=float(config['params']['R21']))
    panel_ply_angles = config['params']['panel_ply_angles'].split()
    panel_ply_thicknesses = config['params']['panel_ply_thicknesses'].split()
    panel_laminate = Laminate([Ply(mat=material, t=float(t), theta=float(theta)) for (theta, t) in zip(panel_ply_angles, panel_ply_thicknesses)], knockdown_factor=float(config['params']['knockdown_factor']),
                              sigma_ult_c=float(config['params']['sigma_ult_c']))
    panel = Panel(laminate=panel_laminate, length=float(config['params']['panel_length']), width=float(config['params']['panel_width']))
    stringer_ply_angles = config['params']['stringer_ply_angles'].split()
    stringer_ply_thicknesses = config['params']['stringer_ply_thicknesses'].split()
    stringer_laminate = Laminate([Ply(mat=material, t=float(t), theta=float(theta)) for (theta, t) in zip(stringer_ply_angles, stringer_ply_thicknesses)], knockdown_factor=float(config['params']['knockdown_factor']),
                                 sigma_ult_c=float(config['params']['sigma_ult_c']))
    return StiffenedPanel(panel=panel, stringer=Stringer(laminate=stringer_laminate, flange_width=float(config['params']['stringer_flange_width']), web_height=float(config['params']['stringer_web_height'])))


def read_fea_results(config: cp.ConfigParser) -> tuple[pd.DataFrame, ...]:
    prepare_csv_for_pandas(config['files']['ply_stress_xx_path'])
    # noinspection PyTypeChecker
    ply_stress_xx: pd.DataFrame = pd.read_csv(config['files']['ply_stress_xx_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    prepare_csv_for_pandas(config['files']['ply_stress_yy_path'])
    # noinspection PyTypeChecker
    ply_stress_yy: pd.DataFrame = pd.read_csv(config['files']['ply_stress_yy_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    prepare_csv_for_pandas(config['files']['ply_stress_xy_path'])
    # noinspection PyTypeChecker
    ply_stress_xy: pd.DataFrame = pd.read_csv(config['files']['ply_stress_xy_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    prepare_csv_for_pandas(config['files']['stringer_strain_path'])
    # noinspection PyTypeChecker
    stringer_strain: pd.DataFrame = pd.read_csv(config['files']['stringer_strain_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    prepare_csv_for_pandas(config['files']['panel_stress_path'])
    # noinspection PyTypeChecker
    panel_stress: pd.DataFrame = pd.read_csv(config['files']['panel_stress_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    prepare_csv_for_pandas(config['files']['stringer_stress_path'])
    # noinspection PyTypeChecker
    stringer_stress: pd.DataFrame = pd.read_csv(config['files']['stringer_stress_path'], skiprows=range(10), na_values=None, keep_default_na=False)

    return ply_stress_xx, ply_stress_yy, ply_stress_xy, stringer_strain, panel_stress, stringer_stress


def fill_ASE_Project2024_task2_1_Template_xxxxxxx(decimals: int) -> None:
    config = read_config()
    super_panel = create_stiffened_panel_from_config(config)
    template = read_ASE_Project2024_task2_1_Template_xxxxxxx(config)
    ply_stress_xx, ply_stress_yy, ply_stress_xy, stringer_strain, panel_stress, stringer_stress = read_fea_results(config)

    UL_factor: float = float(config['params']['ultimate_load_factor'])

    # Enter the ABD matrix of the stringer laminate:
    ABD = super_panel.stringer.laminate.ABD
    template.iloc[16:19, 0:3] = ABD[0:3, 0:3].round(decimals)
    template.iloc[20:23, 0:3] = ABD[0:3, 3:6].round(decimals)
    template.iloc[24:27, 0:3] = ABD[3:6, 3:6].round(decimals)

    # Enter the combined cross-sectional properties for combined column buckling:
    template.iloc[68:72, 1] = round(super_panel.stringer.E_hom_B_x_b_flange, decimals)
    template.iloc[68:72, 2] = round(super_panel.stringer.E_hom_B_x_b_web, decimals)
    template.iloc[68:72, 3:5] = round(super_panel.panel.E_hom_B_x_b, decimals)
    template.iloc[68:72, 5] = round(super_panel.z_EC, decimals)
    template.iloc[68:72, 6] = round(super_panel.EI_hom_B_yy, decimals)
    template.iloc[68:72, 7] = round(super_panel.radius_of_gyration, decimals)
    template.iloc[68:72, 8] = round(super_panel.slenderness, decimals)
    template.iloc[68:72, 9] = round(super_panel.critical_slenderness, decimals)

    # For each load case, enter...

    for LC in range(3):

        # ...RF_FF, RF_IFF, matrix failure mode and RF_strength for element 1:
        for ply in range(8):
            strength_failure: PuckFailure = super_panel.panel.laminate.plies[ply].strength_failure(
                UL_factor * ply_stress_xx[(ply_stress_xx['Elements'] == 1) & (ply_stress_xx['Loadcase'] == LC + 1) & (ply_stress_xx['Layer'] == "Ply  " + str(ply + 1))]["Composite Stresses:Normal X Stress"].tolist()[0],
                UL_factor * ply_stress_yy[(ply_stress_yy['Elements'] == 1) & (ply_stress_yy['Loadcase'] == LC + 1) & (ply_stress_yy['Layer'] == "Ply  " + str(ply + 1))]["Composite Stresses:Normal Y Stress"].tolist()[0],
                UL_factor * ply_stress_xy[(ply_stress_xy['Elements'] == 1) & (ply_stress_xy['Loadcase'] == LC + 1) & (ply_stress_xy['Layer'] == "Ply  " + str(ply + 1))]["Composite Stresses:Shear XY Stress"].tolist()[0])
            template.iat[31 + ply, 1 + 6 * LC] = round(strength_failure.RF_FF, decimals)
            template.iat[31 + ply, 2 + 6 * LC] = round(strength_failure.RF_IFF, decimals)
            template.iat[31 + ply, 3 + 6 * LC] = strength_failure.mode
            template.iat[31 + ply, 4 + 6 * LC] = round(min(strength_failure.RF_FF, strength_failure.RF_IFF), decimals)

        # ...RF_FF, RF_IFF, matrix failure mode and RF_strength for element 40:
        strength_failures: list[PuckFailure] = super_panel.stringer.ply_strength_failures_from_1D_strain(
            UL_factor * stringer_strain[(stringer_strain['Elements'] == 40) & (stringer_strain['Loadcase'] == LC + 1)]['Element Strains (1D):CBAR/CBEAM Axial Strain'].tolist()[0])
        for ply in range(8):
            template.iat[39 + ply, 1 + 6 * LC] = round(strength_failures[ply].RF_FF, decimals)
            template.iat[39 + ply, 2 + 6 * LC] = round(strength_failures[ply].RF_IFF, decimals)
            template.iat[39 + ply, 3 + 6 * LC] = strength_failures[ply].mode
            template.iat[39 + ply, 4 + 6 * LC] = round(min(strength_failures[ply].RF_FF, strength_failures[ply].RF_IFF), decimals)

        # ...sig_xx_avg, sig_yy_avg, sig_xy_avg, sig_crit_shear, sig_crit_biaxial and RF_panel_buckling for all panels:
        for panel in range(5):
            sig_xx_avg = np.average(panel_stress[(panel_stress['Elements'].between(1 + panel * 6, 6 + panel * 6) & (panel_stress['Loadcase'] == LC + 1))]['XX'].tolist())
            sig_yy_avg = np.average(panel_stress[(panel_stress['Elements'].between(1 + panel * 6, 6 + panel * 6) & (panel_stress['Loadcase'] == LC + 1))]['YY'].tolist())
            sig_xy_avg = np.average(panel_stress[(panel_stress['Elements'].between(1 + panel * 6, 6 + panel * 6) & (panel_stress['Loadcase'] == LC + 1))]['XY'].tolist())
            template.iat[52 + panel, 1 + 8 * LC] = round(sig_xx_avg, decimals)
            template.iat[52 + panel, 2 + 8 * LC] = round(sig_yy_avg, decimals)
            template.iat[52 + panel, 3 + 8 * LC] = round(sig_xy_avg, decimals)
            template.iat[52 + panel, 4 + 8 * LC] = round(super_panel.panel.tau_xy_crit_shear, decimals)
            template.iat[52 + panel, 5 + 8 * LC] = round(super_panel.panel.sigma_x_crit_biaxial(sigma_x=sig_xx_avg, sigma_y=sig_yy_avg, m_max=5, n_max=5), decimals)
            template.iat[52 + panel, 6 + 8 * LC] = round(super_panel.panel.RF_panel_buckling(sigma_x=UL_factor * sig_xx_avg, sigma_y=UL_factor * sig_yy_avg, tau_xy=UL_factor * sig_xy_avg, m_max=5, n_max=5), decimals)

        # ...sig_axial_comb_avg, sig_crip, RF_column_buckling_combined for all stiffened panels:
        for stringer in range(4):
            sigma_axial_comb_avg = super_panel.average_stress(panel_stresses=panel_stress[(panel_stress['Elements'].between(4 + stringer * 6, 9 + stringer * 6) & (panel_stress['Loadcase'] == LC + 1))]['XX'].tolist(),
                                                              stringer_stresses=stringer_stress[(stringer_stress['Elements'].between(40 + stringer * 6, 42 + stringer * 6) & (stringer_stress['Loadcase'] == LC + 1))]
                                                              ['Element Stresses (1D):CBAR/CBEAM Axial Stress'].tolist())
            template.iat[61 + stringer, 1 + 5 * LC] = round(sigma_axial_comb_avg, decimals)
            template.iat[61 + stringer, 2 + 5 * LC] = round(super_panel.stringer.sigma_crippling_web, decimals)
            template.iat[61 + stringer, 3 + 5 * LC] = round(-super_panel.critical_buckling_stress / (UL_factor * sigma_axial_comb_avg), decimals)

    # noinspection PyTypeChecker
    template.to_csv(config['files']['template_path'], sep=';', index=False, header=None)


if __name__ == '__main__':
    pass
