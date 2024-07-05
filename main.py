import inout as io


if __name__ == '__main__':

    # Step 1: Calculate the homogenized engineering constants and enter them in "Materials/MAT1/homogenized laminate" in HyperWorks:
    super_panel = io.create_stiffened_panel_from_config(io.read_config())
    print("E = " + str(super_panel.stringer.E_hom_avg_x))
    print("G = " + str(super_panel.stringer.G_hom_avg_xy))

    # Step 2: Fill the submission template:
    # io.fill_ASE_Project2024_task2_1_Template_xxxxxxx(decimals=2)
