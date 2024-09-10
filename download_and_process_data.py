from datetime import datetime, timedelta

from ai import cdas
import numpy as np
import pandas as pd


def download_and_process_data(shock_datetime, SC_ID, filter_options):
    # Select one hour before and after the shock.
    t_start = shock_datetime - timedelta(hours=1)
    t_end = shock_datetime + timedelta(hours=1)

    # Convert the shock time from datetime to seconds after January 1,
    # 1970 (Unix epoch).
    shock_epoch = int(shock_datetime.timestamp())

    if SC_ID == 0:  # ACE
        mag = cdas.get_data(
            'sp_phys', 'AC_H0_MFI', t_start, t_end, ['Magnitude', 'BGSEc'])
        pla = cdas.get_data(
            'sp_phys', 'AC_H0_SWE', t_start, t_end,
            ['Np', 'Vp', 'Tpr', 'V_GSE'])
        pos = cdas.get_data(
            'sp_phys', 'AC_H0_MFI', t_start, t_end, ['SC_pos_GSE'])

        # Magnetic field data
        t_mag = mag['EPOCH'] + timedelta(seconds=8)
       
        B = mag['<|B|>'] 
        Bx = mag['BX_GSE']
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        
        # Plasma data
        t_pla = pla['EPOCH'] + timedelta(seconds=32)
        Np = pla['H_DENSITY']
        V = pla['SW_H_SPEED']
        Tp = pla['H_TEMP_RADIAL']
        Vx = pla['VX_(GSE)']
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)']

        # Position data
        pos_X = pos['ACE_X-GSE']
        pos_Y = pos['ACE_Y-GSE']
        pos_Z = pos['ACE_Z-GSE']
        t_pos = pos['EPOCH']
    # WIND
    if SC_ID == 1:
        mag = cdas.get_data(
            'sp_phys', 'WI_H0_MFI', t_start, t_end, ['B3F1', 'B3GSE'])
        pla = cdas.get_data(
            'sp_phys', 'WI_K0_SWE', t_start, t_end,
            ['Np', 'V_GSE_plog', 'THERMAL_SPD', 'V_GSE'])
        pos = cdas.get_data(
            'sp_phys', 'WI_H0_MFI', t_start, t_end, ['PGSE'])

        # Magnetic field data
        B = mag['B']
        Bx = mag['BX_(GSE)']
        By = mag['BY_(GSE)']
        Bz = mag['BZ_(GSE)'] 
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['ION_NP']
        V = pla['FLOW_SPEED'] 
        v_th = pla['SW_VTH']  
        Vx = pla['VX_(GSE)']
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)'] 
        t_pla = pla['EPOCH']

        # Position data
        pos_X = pos['X_(GSE)']
        pos_Y = pos['Y_(GSE)']
        pos_Z = pos['Z_(GSE)']
        t_pos = pos['EPOCH']

        # Transform thermal speed to temperature
        Tp = 60.57376 * v_th**2

        # Measurement bin radius for each data point
        bin_rad_wind_pla = pla['DEL_TIME'] * 1e-3
    # STEREO-A
    if SC_ID == 2:
        mag = cdas.get_data('sp_phys', 'STA_L1_MAG_RTN', t_start, t_end, ['BFIELD'])
        pla = cdas.get_data('sp_phys', 'STA_L2_PLA_1DMAX_1MIN', t_start, t_end, ['proton_number_density', 'proton_bulk_speed',
                                                                                'proton_temperature', 'proton_Vr_RTN', 
                                                                                'proton_Vt_RTN', 'proton_Vn_RTN'])
        
        pos = cdas.get_data('sp_phys', 'STA_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end, ['radialDistance', 'heliographicLatitude', 'heliographicLongitude'])

        # Magnetic field data
        B = mag['BTOTAL']
        Bx = mag['BR'] #B components
        By = mag['BT']
        Bz = mag['BN'] 
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['DENSITY']
        V = pla['SPEED']
        Tp = pla['TEMPERATURE']
        t_pla = pla['EPOCH'] + timedelta(seconds=30)
        Vx = pla['VR'] #V components
        Vy = pla['VT']
        Vz = pla['VN']

        # Position data
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
        t_pos = pos['EPOCH'] + timedelta(seconds=30*60)


    # STEREO-B
    if SC_ID == 3:
        mag = cdas.get_data('sp_phys', 'STB_L1_MAG_RTN', t_start, t_end, ['BFIELD'])
        pla = cdas.get_data('sp_phys', 'STB_L2_PLA_1DMAX_1MIN', t_start, t_end, ['proton_number_density', 'proton_bulk_speed',
                                                                                'proton_temperature', 'proton_Vr_RTN', 
                                                                                'proton_Vt_RTN', 'proton_Vn_RTN'])
        pos = cdas.get_data('sp_phys', 'STB_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end, ['radialDistance', 'heliographicLatitude', 'heliographicLongitude'])

        # Magnetic field data
        B = mag['BTOTAL']
        Bx = mag['BR'] #B components
        By = mag['BT']
        Bz = mag['BN'] 
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['DENSITY']
        V = pla['SPEED']
        Tp = pla['TEMPERATURE']
        t_pla = pla['EPOCH'] + timedelta(seconds=32)
        Vx = pla['VR'] #V components
        Vy = pla['VT']
        Vz = pla['VN'] 

        # Position data
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
        t_pos = pos['EPOCH'] + timedelta(seconds=30*60)


    # Helios 1
    if SC_ID == 4:

        try:
            mag = cdas.get_data('sp_phys', 'HEL1_6SEC_NESSMAG', t_start, t_end, ['B', 'BXSSE', 'BYSSE', 'BZSSE'])
            HELIOS_no_mag = 0  
        except:                                                   
            HELIOS_no_mag = 1

        pla = cdas.get_data('sp_phys', 'HELIOS1_40SEC_MAG-PLASMA', t_start, t_end, ['B_R', 'B_T', 'B_N', 'Np', 'Tp', 'Vp', 'Vp_R', 'Vp_T', 'Vp_N'])   

        pos = cdas.get_data('sp_phys', 'HELIOS1_40SEC_MAG-PLASMA', t_start, t_end, ['R_Helio', 'clat', 'HGIlong'])
        

        # Plasma data
        Np = pla['N_P']
        V = pla['V_P']
        Tp = pla['T_P']
        Vx = pla['VP_R'] # V components
        Vy = pla['VP_T']
        Vz = pla['VP_N']
        t_pla = pla['EPOCH']

        # Additional magnetic field data from plasma datasets:
        # These will be used later if the primary mag data is faulty.

        # B components from plasma data
        Bx_add1 = pla['B_R']
        By_add1 = pla['B_T']
        Bz_add1 = pla['B_N']
        B_add1 = np.sqrt(Bx_add1**2 + By_add1**2 + Bz_add1**2)
        t_mag_add1 = pla['EPOCH']

        if HELIOS_no_mag == 0:
           
            B = mag['B']
            Bx = -1 * mag['BX_(SSE)'] #B components
            By = -1 * mag['BY_(SSE)']
            Bz = mag['BZ_(SSE)']
            t_mag = mag['EPOCH']
        else:
            
            B = B_add1
            Bx = Bx_add1 #B components
            By = By_add1
            Bz = Bz_add1 
            t_mag = t_mag_add1

        # Position data
        pos_X = pos['R_HELIO']
        pos_Y = pos['CARRINGTON/HGI_LATITUDE']
        pos_Z = pos['HGI_LONGITUDE']
        t_pos = pos['EPOCH']

        

    # Helios 2
    if SC_ID == 5:

        
        try:
            mag = cdas.get_data('sp_phys', 'HEL2_6SEC_NESSMAG', t_start, t_end, ['B', 'BXSSE', 'BYSSE', 'BZSSE'])
    
            HELIOS_no_mag = 0                                                         
        except:
            HELIOS_no_mag = 1

        pla = cdas.get_data('sp_phys', 'HELIOS2_40SEC_MAG-PLASMA', t_start, t_end, ['B_R', 'B_T', 'B_N', 'Np', 'Tp', 'Vp', 
                                                                                    'Vp_R', 'Vp_T', 'Vp_N'])
        pos =  cdas.get_data('sp_phys', 'HELIOS2_40SEC_MAG-PLASMA', t_start, t_end, ['R_Helio', 'clat', 'HGIlong'])
      
       
        # Plasma data
        Np = pla['N_P']
        V = pla['V_P']
        Tp = pla['T_P']
        Vx = pla['VP_R'] # V components
        Vy = pla['VP_T']
        Vz = pla['VP_N']
        t_pla = pla['EPOCH']

        # Additional magnetic field data from plasma datasets:
        # These will be used later if the primary mag data is faulty.

        # B components from plasma data
        Bx_add1 = pla['B_R']
        By_add1 = pla['B_T']
        Bz_add1 = pla['B_N']
        B_add1 = np.sqrt(Bx_add1**2 + By_add1**2 + Bz_add1**2)
        t_mag_add1 = pla['EPOCH']

        if HELIOS_no_mag == 0:
            # Regular mag data products
            B = mag['B']
            Bx = -1 * mag['BX_(SSE)'] #B components
            By = -1 * mag['BY_(SSE)']
            Bz = mag['BZ_(SSE)']
            t_mag = mag['EPOCH']
        else:
            B = B_add1
            Bx = Bx_add1 #B components
            By = By_add1
            Bz = Bz_add1 
            t_mag = t_mag_add1

        # Position data
        pos_X = pos['R_HELIO']
        pos_Y = pos['CARRINGTON/HGI_LATITUDE']
        pos_Z = pos['HGI_LONGITUDE']
        t_pos = pos['EPOCH']


    # Ulysses
    if SC_ID == 6:
        mag = cdas.get_data('sp_phys', 'UY_1SEC_VHM', t_start, t_end, ['B_MAG', 'B_RTN'])
        pla = cdas.get_data('sp_phys', 'UY_M0_BAI', t_start, t_end, ['Density', 'Temperature', 'Velocity'])
        pos = cdas.get_data('sp_phys', 'UY_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end, ['heliocentricDistance', 'heliographicLatitude', 'heliographicLongitude'])
        pos_vec = [pos['ULYSSES_DIST'], pos['HGI_LAT'], pos['HGI_LONG']]

        # Magnetic field data
        B = mag['MAGNETIC_FIELD_MAGNITUDE']
        Bx = mag['R_COMPONENT'] # B components
        By = mag['T_COMPONENT']
        Bz = mag['N_COMPONENT']
        t_mag = mag['EPOCH_TIME']

        # Plasma data
        Np = pla['PROTON']
        t_pla = pla['TIME']
        Vx = pla['VEL_R'] # V components
        Vy = pla['VEL_T']
        Vz = pla['VEL_N'] 
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2) #Bulk speed

        # Calculate the mean of the minimum and maximum value temperature datasets
        Tmax = pla['T-LARGE']
        Tmin = pla['T-SMALL']
        Tp = (Tmax + Tmin) / 2

        # Position data
        pos_X = pos['ULYSSES_DIST']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
        t_pos = pos['EPOCH']

    # Cluster 3
    if SC_ID == 7:
        mag = cdas.get_data('sp_phys', 'C3_CP_FGM_SPIN', t_start, t_end, ['B_mag__C3_CP_FGM_SPIN', 'B_vec_xyz_gse__C3_CP_FGM_SPIN'])
        pla = cdas.get_data('sp_phys', 'C3_PP_CIS', t_start, t_end, ['Status__C3_PP_CIS', 'N_HIA__C3_PP_CIS', 
                                                                    'V_HIA_xyz_gse__C3_PP_CIS', 'T_HIA_par__C3_PP_CIS', 
                                                                    'T_HIA_perp__C3_PP_CIS'])
        pos = cdas.get_data('sp_phys', 'C3_CP_FGM_SPIN', t_start, t_end, ['sc_pos_xyz_gse__C3_CP_FGM_SPIN'])
        
        # Magnetic field data
        B = mag['B']
        Bx = mag['BX_GSE'] #B components
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        t_mag = mag['UT']

        # Plasma data
        status = pla['STATUS[1]'] # Later this variable will be used to check if data was okay
        Np = pla['N(HIA)']
        Tp_par = pla['T(HIA)_PAR']
        Tp_perp = pla['T(HIA)_PERP']
        Vx = pla['VX_HIA_GSE'] #V components
        Vy = pla['VY_HIA_GSE'] 
        Vz = pla['VZ_HIA_GSE']
        t_pla = pla['EPOCH']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2) #Bulk speed

        # Total proton temperature assuming isotropic velocity distribution
        Tp = 2.0/3.0 * Tp_perp + 1.0/3.0 * Tp_par
        Tp *= 1.0e6 #Unit conversion from MK to K

        # Position data
        pos_X = pos['C3_X_GSE']
        pos_Y = pos['C3_Y_GSE']
        pos_Z = pos['C3_Z_GSE']
        t_pos = pos['UT'] 

    # Cluster 1
    if SC_ID == 8:
        mag = cdas.get_data('sp_phys', 'C1_CP_FGM_SPIN', t_start, t_end, ['B_mag__C1_CP_FGM_SPIN', 'B_vec_xyz_gse__C1_CP_FGM_SPIN'])
        pla = cdas.get_data('sp_phys', 'C1_PP_CIS', t_start, t_end, ['Status__C1_PP_CIS', 'N_HIA__C1_PP_CIS', 
                                                                    'V_HIA_xyz_gse__C1_PP_CIS', 'T_HIA_par__C1_PP_CIS', 
                                                                    'T_HIA_perp__C1_PP_CIS'])
        pos = cdas.get_data('sp_phys', 'C1_CP_FGM_SPIN', t_start, t_end, ['sc_pos_xyz_gse__C1_CP_FGM_SPIN'])
        

        # Magnetic field data
        B = mag['B']
        Bx = mag['BX_GSE']  #B components
        By = mag['BY_GSE'] 
        Bz = mag['BZ_GSE'] 
        t_mag = mag['UT']

        # Plasma data
        status = pla['STATUS[1]'] # Later this variable will be used to check if data was okay
        Np = pla['N(HIA)']
        Tp_par = pla['T(HIA)_PAR']
        Tp_perp = pla['T(HIA)_PERP']
        t_pla = pla['EPOCH']
        Vx = pla['VX_HIA_GSE'] #V components
        Vy = pla['VY_HIA_GSE']
        Vz = pla['VZ_HIA_GSE']
        
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2) #Bulk speed

        # Total proton temperature assuming isotropic velocity distribution
        Tp = 2.0/3.0 * Tp_perp + 1.0/3.0 * Tp_par
        Tp *= 1.0e6 #Unit conversion from MK to K

        pos_X = pos['C1_X_GSE']
        pos_Y = pos['C1_Y_GSE']
        pos_Z = pos['C1_Z_GSE']
        t_pos = pos['UT'] 
    
    # Cluster 4
    if SC_ID == 9:
        mag = cdas.get_data('sp_phys', 'C4_CP_FGM_SPIN', t_start, t_end, ['B_mag__C4_CP_FGM_SPIN', 'B_vec_xyz_gse__C4_CP_FGM_SPIN'])
        pla = cdas.get_data('sp_phys', 'C4_PP_CIS', t_start, t_end, ['Status__C4_PP_CIS', 'N_p__C4_PP_CIS', 
                                                                    'V_p_xyz_gse__C4_PP_CIS', 'T_p_par__C4_PP_CIS', 
                                                                    'T_p_perp__C4_PP_CIS'])
        pos = cdas.get_data('sp_phys', 'C4_CP_FGM_SPIN', t_start, t_end, ['sc_pos_xyz_gse__C4_CP_FGM_SPIN'])
    

        # Magnetic field data
        B = mag['B']
        Bx = mag['BX_GSE'] #B components
        By = mag['BY_GSE'] 
        Bz = mag['BZ_GSE'] 
        t_mag = mag['UT'] 

        # Plasma data
        status = pla['STATUS[1]'] # Later this variable will be used to check if data was okay
        Np = pla['N(P)']
        Tp_par = pla['T(P)_PAR']
        Tp_perp = pla['T(P)_PERP']
        Vx = pla['VX_P_GSE'] #V components
        Vy = pla['VY_P_GSE'] 
        Vz = pla['VZ_P_GSE']
        t_pla = pla['EPOCH']

        V = np.sqrt(Vx**2 + Vy**2 + Vz**2) #Bulk speed

        # Total proton temperature assuming isotropic velocity distribution
        Tp = 2.0/3.0 * Tp_perp + 1.0/3.0 * Tp_par
        Tp *= 1.0e6 # Unit conversion from MK to K

        # Position data
        pos_X = pos['C4_X_GSE']
        pos_Y = pos['C4_Y_GSE']
        pos_Z = pos['C4_Z_GSE']
        t_pos = pos['UT']

    # OMNI
    if SC_ID == 10:
        mag = cdas.get_data('sp_phys', 'OMNI_HRO_1MIN', t_start, t_end, ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE'])
        pla = cdas.get_data('sp_phys', 'OMNI_HRO_1MIN', t_start, t_end, ['flow_speed', 'Vx', 'Vy', 'Vz', 'proton_density', 'T'])
        pos = cdas.get_data('sp_phys', 'OMNI_HRO_1MIN', t_start, t_end, ['BSN_x', 'BSN_y', 'BSN_z'])
        pos_vec = [pos['X_(BSN),_GSE'], pos['Y_(BSN),_GSE'], pos['Z_(BSN),_GSE']]
        t_pos = pos['EPOCH_TIME']

        # Magnetic field data
        B = mag['MAG_AVG_B-VECTOR']
        Bx = mag['BX,_GSE'] #B components
        By = mag['BY,_GSE']
        Bz = mag['BZ,_GSE']
        t_mag = mag['EPOCH_TIME']

        # Plasma data
        Np = pla['PROTON_DENSITY']
        Tp = pla['TEMPERATURE']
        V = pla['FLOW_SPEED,_GSE'] 
        Vx = pla['VX_VELOCITY,_GSE'] #V components
        Vy = pla['VY_VELOCITY,_GSE']
        Vz = pla['VZ_VELOCITY,_GSE']     
        t_pla = pla['EPOCH_TIME']

        # Position data
        pos_X = pos['X_(BSN),_GSE']
        pos_Y = pos['Y_(BSN),_GSE']
        pos_Z = pos['Z_(BSN),_GSE']
        t_pos = pos['EPOCH_TIME']

    # Voyager 1
    if SC_ID == 11:
        mag = cdas.get_data('sp_phys', 'VOYAGER1_2S_MAG', t_start, t_end, ['F1', 'B1', 'B2', 'B3'])
        pla = cdas.get_data('sp_phys', 'VOYAGER1_PLS_HIRES_PLASMA_DATA', t_start, t_end, ['V', 'V_rtn', 'dens', 'V_thermal'])
        pos = cdas.get_data('sp_phys', 'VOYAGER1_2S_MAG', t_start, t_end, ['scDistance', 'scLon', 'scLat'])
        
        # Magnetic field data
        B = mag['B-MAGNITUDE_(F1)']
        Bx = mag['BR_(B1)'] #B components
        By = mag['BT_(B2)']
        Bz = mag['BN_(B3)']
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['PROTON_DENSITY']
        V_th = pla['VP_THERMAL']
        Tp = 60.57376 * V_th**2
        V = pla['VP']
        Vx = pla['VR'] #V components
        Vy = pla['VT']
        Vz = pla['VN']
        t_pla = pla['EPOCH']

        rad2deg = 180.0 / np.pi  # Conversion factor from radians to degrees
        pos_X = pos['V1_DIST']
        pos_Y = pos['V1_LAT_IHG'] * rad2deg
        pos_Z = pos['V1_LONG_IHG'] * rad2deg
        t_pos = pos['EPOCH']

    # Voyager 2
    if SC_ID == 12:
        mag = cdas.get_data('sp_phys', 'VOYAGER2_2S_MAG', t_start, t_end, ['F1', 'B1', 'B2', 'B3'])
        pla = cdas.get_data('sp_phys', 'VOYAGER2_PLS_HIRES_PLASMA_DATA', t_start, t_end, ['V', 'V_rtn', 'dens', 'V_thermal'])
        pos = cdas.get_data('sp_phys', 'VOYAGER1_2S_MAG', t_start, t_end, ['scDistance', 'scLon', 'scLat'])
    
        # Magnetic field data
        B = mag['B-MAGNITUDE_(F1)']
        Bx = mag['BR_(B1)'] #B components
        By = mag['BT_(B2)']
        Bz = mag['BN_(B3)']
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['PROTON_DENSITY']
        V_th = pla['VP_THERMAL']
        Tp = 60.57376 * V_th**2
        V = pla['VP']
        Vx = pla['VR'] #V components
        Vy = pla['VT']
        Vz = pla['VN']
        t_pla = pla['EPOCH']

        rad2deg = 180.0 / np.pi  # Conversion factor from radians to degrees
        pos_X = pos['V1_DIST']
        pos_Y = pos['V1_LAT_IHG'] * rad2deg
        pos_Z = pos['V1_LONG_IHG'] * rad2deg
        t_pos = pos['EPOCH']

    # DSCOVR
    if SC_ID == 13:
        mag = cdas.get_data('sp_phys', 'DSCOVR_H0_MAG', t_start, t_end, ['B1F1', 'B1GSE'])
        pla = cdas.get_data('sp_phys', 'DSCOVR_H1_FC', t_start, t_end, ['V_GSE', 'Np', 'THERMAL_TEMP'])
        pos = cdas.get_data('sp_phys', 'DSCOVR_ORBIT_PRE', t_start, t_end, ['GSE_POS'])

        # Magnetic field data
        B = mag['B']
        Bx = mag['BX_(GSE)'] #B components
        By = mag['BY_(GSE)']
        Bz = mag['BZ_(GSE)']
        t_mag = mag['EPOCH']

        # Plasma data
        Np = pla['ION_N']
        Tp = pla['TEMPERATURE']
        Vx = pla['VX_(GSE)'] #V components
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)'] 
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        t_pla=pla['EPOCH'] 

        # Position data
        pos_X = pos['X_GSE']
        pos_Y = pos['Y_GSE']
        pos_Z = pos['Z_GSE']
        t_pos = pos['EPOCH']

    #PSP
    if SC_ID == 14:
        mag = cdas.get_data('sp_phys', 'PSP_FLD_L2_MAG_RTN_1MIN', t_start, t_end, ["psp_fld_l2_mag_RTN_1min"])
        pla = cdas.get_data('sp_phys', 'PSP_SWP_SPC_L3I', t_start, t_end, ["np_moment_gd", "vp_moment_RTN_gd", 'wp_moment'])
        pos2 = cdas.get_data('sp_phys', 'PSP_SWP_SPC_L3I', t_start, t_end, ['sc_pos_HCI'])

        # Magnetic field data
        Bx = mag['B_R'] #B components
        By = mag['B_T']
        Bz = mag['B_N']
        t_mag = mag['EPOCH']
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

        # Plasma data
        Np = pla['NP_MOMENT']
        Tp = pla['WP_MOMENT'] #double check

        #V_th = pla['W3_FIT']    #w3_fit_gd? Most probable thermal speed. or wp1_fit? 
        #Tp = 60.57376 * V_th**2 # Is this right ?

        Vx = pla['VP_MOMENT_R'] #V components
        Vy = pla['VP_MOMENT_T']
        Vz = pla['VP_MOMENT_N']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        t_pla=pla['EPOCH']

        # Position data - HCI
        pos_X = pos2['X_HCI']
        pos_Y = pos2['Y_HCI']
        pos_Z = pos2['Z_HCI']
        t_pos = pos2['EPOCH']

        #Converting HCI to (rad, lat, long format)
        rad_dist = np.sqrt(pos_X**2 + pos_Y**2 + pos_Z**2) * 6.68458712 * 10**-9
        lat = np.rad2deg(np.arctan(pos_Z / np.sqrt(pos_X**2 + pos_Y**2)))
        long = np.mod(np.rad2deg(np.arctan2(pos_Y, pos_X)), 360)

        pos_X = rad_dist
        pos_Y = lat
        pos_Z = long


    #SOlo
    if SC_ID == 15:
        mag = cdas.get_data('sp_phys', 'SOLO_L2_MAG-RTN-NORMAL', t_start, t_end, ["B_RTN"])
        pla = cdas.get_data('sp_phys', 'SOLO_L2_SWA-PAS-GRND-MOM', t_start, t_end, ["N", "V_RTN", "T"])
        pos = cdas.get_data('sp_phys', 'SOLO_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end, ["radialDistance", "heliographicLatitude", "heliographicLongitude"])

        # Magnetic field data
        Bx = mag['B_R'] #B components
        By = mag['B_T']
        Bz = mag['B_N']
        t_mag = mag['EPOCH']
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

        # Plasma data
        Np = pla['DENSITY']
        Tp = pla['TEMPERATURE'] * 11604.52500617
        Vx = pla['VR_RTN'] #V components
        Vy = pla['VT_RTN']
        Vz = pla['VN_RTN']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        t_pla=pla['EPOCH']

        # Position data
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
        t_pos = pos['EPOCH']
    
##------------------------------------------------------------------------------
## Create pandas dataframe of the dataproducts
##------------------------------------------------------------------------------
    if SC_ID not in [7, 8, 9]:
        status = np.full(len(t_pla), np.nan)
        
    #Converting all time vectors from datetime format to UNIX epoch meaning seconds since 1st Jan 1970
    t_mag = [dt.timestamp() for dt in t_mag]
    t_pla = [dt.timestamp() for dt in t_pla]
    t_pos = [dt.timestamp() for dt in t_pos]
    
    # Create dataframes
    mag_dataframe = pd.DataFrame({
        'EPOCH': t_mag,
        'B': B,
        'Bx': Bx,
        'By': By,
        'Bz': Bz
    })

    pla_dataframe = pd.DataFrame({
        'EPOCH': t_pla,
        'Np': Np,
        'V': V,
        'Tp': Tp,
        'Vx': Vx,
        'Vy': Vy,
        'Vz': Vz,
        'status': status
    })

    pos_dataframe = pd.DataFrame({
        'EPOCH': t_pos,
        'pos_X': pos_X,
        'pos_Y': pos_Y,
        'pos_Z': pos_Z
    })
    
    # Additional dataframe for helios
    if SC_ID == 4 or SC_ID == 5:
        add_dataframe = pd.DataFrame({
            'EPOCH': [t.timestamp() for t in t_mag_add1],
            'B': B_add1,
            'Bx': Bx_add1,
            'By': By_add1,
            'Bz': Bz_add1
            })
    else:
        add_dataframe = pd.DataFrame()

#------------------------------------------------------------------------------
# Cleaning the data (removing bad values)
#------------------------------------------------------------------------------

    # Magnetic field magnitude data
    mag_dataframe['B'] = mag_dataframe['B'].mask((mag_dataframe['B'] > 150) | (mag_dataframe['B'] <= 0))

    # Magnetic field components
    mag_dataframe['Bx'] = mag_dataframe['Bx'].mask(np.abs(mag_dataframe['Bx']) > 1500)
    mag_dataframe['By'] = mag_dataframe['By'].mask(np.abs(mag_dataframe['By']) > 1500)
    mag_dataframe['Bz'] = mag_dataframe['Bz'].mask(np.abs(mag_dataframe['Bz']) > 1500)

    # Solar wind bulk speed
    pla_dataframe['V'] = pla_dataframe['V'].mask((pla_dataframe['V'] > 1500) | (pla_dataframe['V'] < 0))

    # Solar wind velocity components
    pla_dataframe['Vx'] = pla_dataframe['Vx'].mask(np.abs(pla_dataframe['Vx']) > 1500)
    pla_dataframe['Vy'] = pla_dataframe['Vy'].mask(np.abs(pla_dataframe['Vy']) > 1500)
    pla_dataframe['Vz'] = pla_dataframe['Vz'].mask(np.abs(pla_dataframe['Vz']) > 1500)

    # Density
    pla_dataframe['Np'] = pla_dataframe['Np'].mask((pla_dataframe['Np'] > 150) | (pla_dataframe['Np'] < 0))

    # Temperature
    pla_dataframe['Tp'] = pla_dataframe['Tp'].mask((pla_dataframe['Tp'] > 0.9e7) | (pla_dataframe['Tp'] < 0))

    # Position data
    pos_dataframe['pos_X'] = pos_dataframe['pos_X'].mask(np.abs(pos_dataframe['pos_X']) > 1e9)
    pos_dataframe['pos_Y'] = pos_dataframe['pos_Y'].mask(np.abs(pos_dataframe['pos_Y']) > 1e9)
    pos_dataframe['pos_Z'] = pos_dataframe['pos_Z'].mask(np.abs(pos_dataframe['pos_Z']) > 1e9)

    # Deleting values where the instrument status was wrong (5 or higher)
    # This only applies for Cluster SC
    if SC_ID in [7, 8, 9]:
        status = pla_dataframe['status']
        faulty_values = status > 5
        pla_dataframe['V'] = pla_dataframe['V'].mask(faulty_values)
        pla_dataframe['Vx'] = pla_dataframe['Vx'].mask(faulty_values)
        pla_dataframe['Vy'] = pla_dataframe['Vy'].mask(faulty_values)
        pla_dataframe['Vz'] = pla_dataframe['Vz'].mask(faulty_values)
        pla_dataframe['Np'] = pla_dataframe['Np'].mask(faulty_values)
        pla_dataframe['Tp'] = pla_dataframe['Tp'].mask(faulty_values)
        pla_dataframe = pla_dataframe.drop(columns=['status'])

    # Cleaning the additional helios dataframe
    if SC_ID == 4 or SC_ID == 5:
        condition = (add_dataframe['B'] > 150) | (add_dataframe['B'] <= 0)
        add_dataframe['B'] = add_dataframe['B'].mask(condition)
        add_dataframe['Bx'] = add_dataframe['Bx'].mask(condition)
        add_dataframe['By'] = add_dataframe['By'].mask(condition)
        add_dataframe['Bz'] = add_dataframe['Bz'].mask(condition)

        condition = (add_dataframe['Bx'] > 1500) | (add_dataframe['By'] > 1500) | (add_dataframe['By'] > 1500)
        add_dataframe['B'] = add_dataframe['B'].mask(condition)
        add_dataframe['Bx'] = add_dataframe['Bx'].mask(condition)
        add_dataframe['By'] = add_dataframe['By'].mask(condition)
        add_dataframe['Bz'] = add_dataframe['Bz'].mask(condition)

# ----------------------------------------------------------------------
# Median filter for data spikes
# ----------------------------------------------------------------------

    if isinstance(filter_options, int):
        filter_options = [filter_options]

    # If filter option was chosen:
    if filter_options[0] != 0:

        # User-defined settings
        if len(filter_options) == 4:    
            median_window_size = filter_options[0]
            tols = filter_options[1:4]

        # Default settings
        elif filter_options[0] == 1:
            median_window_size = 5
            tols = [0.75, 1.5, 0.2]

        # Incorrect input            
        else:
            raise ValueError('Incorrect input for the spike filter. Must be either 0, 1, or 4-element float array')

        # Taking a mean of Np, Tp and V dataseries and removing values that differ
        # from the mean series too much. Allowed difference is set by the corresponding
        # multiplier in the tols array. Np = tols[1], Tp - tols[2] and V - tols[3]
    
        for col_name in ['Np', 'Tp', 'V']:
            

            pla_dataframe[col_name + '_median'] = pla_dataframe[col_name].rolling(window=int(median_window_size), center=True).median()

            # Calculate the absolute difference from the median
            diff = np.abs(pla_dataframe[col_name] - pla_dataframe[col_name + '_median'])

            # Calculate the tolerance limit based on the multiplier
            tol_limit = tols[['Np', 'Tp', 'V'].index(col_name)] * pla_dataframe[col_name + '_median']

            # Replace values exceeding the tolerance with NaN
            pla_dataframe.loc[diff > tol_limit, col_name] = np.nan

        # Find indices where V contains NaN values
        missing_velocity = pla_dataframe['V'].isna()

        # Set corresponding elements in Vx, Vy, and Vz to NaN
        pla_dataframe.loc[missing_velocity, ['Vx', 'Vy', 'Vz']] = np.nan


##------------------------------------------------------------------------------
## If the given shock time is only preliminary, a better estimate is determined
## using magnetic field data
##------------------------------------------------------------------------------

    # THIS FEATURE HAS BEEN TURNED OFF AS SHOCK TIME IS ADJUSTED WITH LEFT AND RIGHT ARROWS

    # Preliminary (t_pre == 1) / Not Preliminary (t_pre == 0)    
    #if t_pre == 1:
    #    t_shock = check_shock_time(mag_dataframe['EPOCH'], mag_dataframe['B'], shock_epoch)

        # Additional shock time estimate based on additional Helios magnetic 
        # Field data
        #if (SC_ID == 4) or (SC_ID == 5) and (HELIOS_no_mag == 0):
        #    t_shock_new_add1 = check_shock_time(add_dataframe['EPOCH'], add_dataframe['B'], shock_epoch)
        #else:
        #    t_shock_new_add1 = np.nan
        #
        #t_shock_new = t_shock

    
        #if time is not preliminary, new time is the same 
    t_shock_new = shock_epoch
    t_shock_new_add1 = shock_epoch
        
    
##------------------------------------------------------------------------------
## Collecting the position data around the time of the shock
##------------------------------------------------------------------------------

    # Find the index of the closest time point to the shock_epoch
    idx = np.argmin(np.abs(pos_dataframe['EPOCH'] - shock_epoch))

    # Extract the position vector at the closest time point
    SC_pos = pos_dataframe.loc[idx, ['pos_X', 'pos_Y', 'pos_Z']]

    #Additional position vector based additional Helios data
    if (SC_ID == 4) or (SC_ID == 5) and (HELIOS_no_mag == 0):
        idx = np.argmin(np.abs(add_dataframe['EPOCH'] - shock_epoch))
        SC_pos_add1 = pos_dataframe.loc[idx, ['pos_X', 'pos_Y', 'pos_Z']]

    # Position of ACE, Cluster and DSCOVR spacecraft is in km hence rescaling is needed.
    scaling_factor = 6378.14
    if SC_ID in [0, 7, 8, 9, 13]:
        SC_pos /= scaling_factor
    
##------------------------------------------------------------------------------
## Determining the validity of the position vector of SC at the time of shock 
## (POSITION DATA IS RARELY INVALID. DO NOT CHANGE THIS SECTION UNLESS AN 
##  ADDITIONAL DATA SOURCE TO REPLACE THE INVALID POSITION DATA CAN BE 
##  DETERMINED AND UTILISED)
##------------------------------------------------------------------------------

    # Check if the position vector is valid
    # pos_ch: (valid = 1, invalid = 0)
    
    #pos_ch = 1
    #faulty_valuesp = np.where(np.isfinite(SC_pos))
    #if len(faulty_valuesp[0]) < 3:
    #   pos_ch = 0
    
    #If position vector is not valid, other data sources are used (if possible)

    #ACE has one additional source
    #if (SC_ID == 0) and (pos_ch == 0):
    #    pos, pos_ch = safe_cdaweb_download('AC_H0_SWE', ['SC_pos_GSE'], interval)
    #
    #    pos_vec = [pos['ACE_X-GSE'], pos['ACE_Y-GSE'], pos['ACE_Z-GSE']]
    #    pos_vec = np.array(pos_vec)
    #    t_pos = pos['EPOCH']
    #    t_pos = [dt.timestamp() for dt in t_pos]
    #    t_pos = np.array(t_pos)
        
    #    idx = np.argmin(np.abs(shock_epoch - t_pos))
    #    SC_pos = pos_vec[:, idx] / scaling_factor  # Scale units

    #Wind has two additional sources
    #if (SC_ID == 1) and (pos_ch == 0):

        #first source
        #pos, pos_ch = safe_cdaweb_download('WI_K0_SWE', ['SC_pos_gse'], interval)
        #pos_vec = pos.SC_pos_gse.dat
        #t_pos = pos.epoch.dat
        #idx = min(abs(shock_epoch - t_pos), ind)
        #SC_pos = pos_vec(*,ind)        

        #validity check
        #pos_ch = 1
        #faulty_valuesp = where(finite(SC_pos) eq 1)
        #if n_elements(faulty_valuesp) lt 3 then pos_ch = 0

        #second source (used if first source does not give a valid result)
        #if pos_ch == 0:
            #restrictions in the second additional position data source: 
            #used source depends on the time (before or after 1.7.1997 23:50)
            #CDF_EPOCH, wind_limit, 1997, 07, 01, 23, 50, /COMPUTE_EPOCH
            #if shock_epoch GE wind_limit then begin
            #    pos_title = 'WI_OR_PRE'
            #endif else begin
            #    pos_title = 'WI_OR_DEF'
            #endelse
            #pos, pos_ch = safe_cdaweb_download(pos_title,['GSE_POS'], interval)
            #pos_vec = pos.GSE_POS.dat    
            #t_pos = pos.epoch.dat
            #idx = min(abs(shock_epoch - t_pos), ind)
            #SC_pos = pos_vec(*,ind)        
        
        #SC_pos = SC_pos/scaling_factor #scale units


##------------------------------------------------------------------------------
## The output
##------------------------------------------------------------------------------

    # Additional Helios output
    if (SC_ID == 4) or (SC_ID == 5) and (HELIOS_no_mag == 0):
        output_add = [add_dataframe, t_shock_new_add1, SC_pos_add1]
    else:
        output_add = [add_dataframe, t_shock_new, SC_pos]
        
    # Information of the measurement radiuses is given as an output 
    # (this only applies to WIND SC)
    if SC_ID == 1:
        pla_bin_rads = bin_rad_wind_pla
    else:
        pla_bin_rads = 0

    return mag_dataframe, pla_dataframe, SC_pos, output_add, t_shock_new, pla_bin_rads

    
 