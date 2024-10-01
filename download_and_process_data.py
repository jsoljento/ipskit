from datetime import datetime, timedelta

from ai import cdas
import numpy as np
import pandas as pd
import scipy.constants as constants


def thermal_speed_to_temperature(V_th):
    r"""Convert thermal speed into temperature.

    This functions calculates temperature from thermal speed using
    the formula
    .. math::
        T_{\mathrm{p}} = m_{\mathrm{p}}/(2k_{\mathrm{B}})
            V_{\mathrm{th}}^{2},
    where :math:`m_{\mathrm{p}}` is the mass of a proton and
    :math:`k_{\mathrm{B}}` is the Boltzmann constant.

    Parameters
    ----------
    V_th : float or array_like
        Thermal speed in km/s.

    Returns
    -------
    T_p
        Proton temperature in Kelvins.
    """

    k_B = constants.Boltzmann
    m_p = constants.m_p

    # V_th is multiplied by 1e3 to convert from km/s to m/s.
    T_p = m_p / (2 * k_B) * (1e3 * V_th)**2

    return T_p


def download_and_process_data(shock_datetime, sc, filter_options):
    """Download and process spacecraft data.

    This functions downloads and processes spacecraft data of a shock.
    The function downloads one hour of data on either side of the shock,
    stores the data in pandas DataFrames, cleans the data up and, if
    the user wants, filter out bad data spikes. The function also
    downloads spacecraft position data. For the Wind spacecraft the
    function also returns the plasma data bin radii.

    Parameters
    ----------
    shock_datetime : datetime
        Time of the shock as a datetime object.
    sc : int
        Spacecraft ID number. See documentation for details.
    filter_options : array_like
        Options for filtering bad data spikes, 0 for no filtering,
        1 for filtering with default values, and a 4-element array
        for filtering with custom values. See documentation for
        details.

    Returns
    -------
    mag_dataframe : DataFrame
        Magnetic field data.
    pla_dataframe : DataFrame
        Plasma data (velocity, speed, density, temperature).
    sc_pos : DataFrame
        Spacecraft position data.
    output_add : list
        Additional Helios output.
    shock_timestamp : int
        Shock time as seconds since January 1, 1970 (Unix time).
    pla_bin_rads : array_like
        Plasma data bin radii. Only returned for the Wind spacecraft.
    """

    # Select one hour before and after the shock
    t_start = shock_datetime - timedelta(hours=1)
    t_end = shock_datetime + timedelta(hours=1)

    # Convert the shock time from datetime to seconds after January 1,
    # 1970 (Unix time)
    shock_timestamp = int(shock_datetime.timestamp())

    # Create empty arrays for the different data products; these will
    # be replaced by the full arrays when the data is downloaded.
    t_mag = np.array([])
    Bx = np.array([])
    By = np.array([])
    Bz = np.array([])
    B = np.array([])

    t_mag_add = np.array([])
    Bx_add = np.array([])
    By_add = np.array([])
    Bz_add = np.array([])
    B_add = np.array([])

    t_pla = np.array([])
    Vx = np.array([])
    Vy = np.array([])
    Vz = np.array([])
    V = np.array([])
    Np = np.array([])
    Tp = np.array([])
    status = np.array([])

    t_pos = np.array([])
    pos_X = np.array([])
    pos_Y = np.array([])
    pos_Z = np.array([])

    # ------------------------------------------------------------------
    # Downloading and preprocessing the data (e.g., converting to
    # correct units)
    # ------------------------------------------------------------------

    if sc == 0:  # ACE
        mag = cdas.get_data(
            'sp_phys', 'AC_H0_MFI', t_start, t_end, ['Magnitude', 'BGSEc'])
        pla = cdas.get_data(
            'sp_phys', 'AC_H0_SWE', t_start, t_end,
            ['Np', 'Vp', 'Tpr', 'V_GSE'])
        pos = cdas.get_data(
            'sp_phys', 'AC_H0_MFI', t_start, t_end, ['SC_pos_GSE'])

        # Magnetic field data
        t_mag = mag['EPOCH'] + timedelta(seconds=8)
        Bx = mag['BX_GSE']
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        B = mag['<|B|>']

        # Plasma data
        t_pla = pla['EPOCH'] + timedelta(seconds=32)
        Vx = pla['VX_(GSE)']
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)']
        V = pla['SW_H_SPEED']
        Np = pla['H_DENSITY']
        Tp = pla['H_TEMP_RADIAL']

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['ACE_X-GSE']
        pos_Y = pos['ACE_Y-GSE']
        pos_Z = pos['ACE_Z-GSE']
    elif sc == 1:  # WIND
        mag = cdas.get_data(
            'sp_phys', 'WI_H0_MFI', t_start, t_end, ['B3F1', 'B3GSE'])
        pla = cdas.get_data(
            'sp_phys', 'WI_K0_SWE', t_start, t_end,
            ['Np', 'V_GSE_plog', 'THERMAL_SPD', 'V_GSE'])
        pos = cdas.get_data(
            'sp_phys', 'WI_H0_MFI', t_start, t_end, ['PGSE'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BX_(GSE)']
        By = mag['BY_(GSE)']
        Bz = mag['BZ_(GSE)']
        B = mag['B']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VX_(GSE)']
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)']
        V = pla['FLOW_SPEED']
        Np = pla['ION_NP']
        V_th = pla['SW_VTH']

        # Convert thermal speed to temperature
        Tp = thermal_speed_to_temperature(V_th)

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['X_(GSE)']
        pos_Y = pos['Y_(GSE)']
        pos_Z = pos['Z_(GSE)']

        # Measurement bin radius for each data point
        bin_rad_wind_pla = pla['DEL_TIME'] * 1e-3
    elif sc == 2:  # STEREO-A
        mag = cdas.get_data(
            'sp_phys', 'STA_L1_MAG_RTN', t_start, t_end, ['BFIELD'])
        pla = cdas.get_data(
            'sp_phys', 'STA_L2_PLA_1DMAX_1MIN', t_start, t_end,
            ['proton_number_density', 'proton_bulk_speed',
             'proton_temperature', 'proton_Vr_RTN', 'proton_Vt_RTN',
             'proton_Vn_RTN'])
        pos = cdas.get_data(
            'sp_phys', 'STA_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end,
            ['radialDistance', 'heliographicLatitude',
             'heliographicLongitude'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BR']
        By = mag['BT']
        Bz = mag['BN']
        B = mag['BTOTAL']

        # Plasma data
        t_pla = pla['EPOCH'] + timedelta(seconds=30)
        Vx = pla['VR']
        Vy = pla['VT']
        Vz = pla['VN']
        V = pla['SPEED']
        Np = pla['DENSITY']
        Tp = pla['TEMPERATURE']

        # Position data
        t_pos = pos['EPOCH'] + timedelta(seconds=30*60)
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
    elif sc == 3:  # STEREO-B
        mag = cdas.get_data(
            'sp_phys', 'STB_L1_MAG_RTN', t_start, t_end, ['BFIELD'])
        pla = cdas.get_data(
            'sp_phys', 'STB_L2_PLA_1DMAX_1MIN', t_start, t_end,
            ['proton_number_density', 'proton_bulk_speed',
             'proton_temperature', 'proton_Vr_RTN',  'proton_Vt_RTN',
             'proton_Vn_RTN'])
        pos = cdas.get_data(
            'sp_phys', 'STB_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end,
            ['radialDistance', 'heliographicLatitude',
             'heliographicLongitude'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BR']
        By = mag['BT']
        Bz = mag['BN']
        B = mag['BTOTAL']

        # Plasma data
        t_pla = pla['EPOCH'] + timedelta(seconds=32)
        Vx = pla['VR']
        Vy = pla['VT']
        Vz = pla['VN']
        V = pla['SPEED']
        Np = pla['DENSITY']
        Tp = pla['TEMPERATURE']

        # Position data
        t_pos = pos['EPOCH'] + timedelta(seconds=30*60)
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
    elif sc == 4:  # Helios 1
        try:
            mag = cdas.get_data(
                'sp_phys', 'HEL1_6SEC_NESSMAG', t_start, t_end,
                ['B', 'BXSSE', 'BYSSE', 'BZSSE'])
            helios_no_mag = 0
        except cdas.NoDataError:
            mag = {}
            helios_no_mag = 1
        pla = cdas.get_data(
            'sp_phys', 'HELIOS1_40SEC_MAG-PLASMA', t_start, t_end,
            ['B_R', 'B_T', 'B_N',
             'Np', 'Tp', 'Vp',
             'Vp_R', 'Vp_T', 'Vp_N'])
        pos = cdas.get_data(
            'sp_phys', 'HELIOS1_40SEC_MAG-PLASMA', t_start, t_end,
            ['R_Helio', 'clat', 'HGIlong'])

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VP_R']
        Vy = pla['VP_T']
        Vz = pla['VP_N']
        V = pla['V_P']
        Np = pla['N_P']
        Tp = pla['T_P']

        # Additional magnetic field data from plasma datasets:
        # These will be used later if the primary mag data is faulty
        t_mag_add = pla['EPOCH']
        Bx_add = pla['B_R']
        By_add = pla['B_T']
        Bz_add = pla['B_N']
        B_add = np.sqrt(Bx_add**2 + By_add**2 + Bz_add**2)

        if helios_no_mag == 0:
            t_mag = mag['EPOCH']
            Bx = -1 * mag['BX_(SSE)']
            By = -1 * mag['BY_(SSE)']
            Bz = mag['BZ_(SSE)']
            B = mag['B']
        else:
            t_mag = t_mag_add
            Bx = Bx_add
            By = By_add
            Bz = Bz_add
            B = B_add

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['R_HELIO']
        pos_Y = pos['CARRINGTON/HGI_LATITUDE']
        pos_Z = pos['HGI_LONGITUDE']
    elif sc == 5:  # Helios 2
        try:
            mag = cdas.get_data(
                'sp_phys', 'HEL2_6SEC_NESSMAG', t_start, t_end,
                ['B', 'BXSSE', 'BYSSE', 'BZSSE'])
            helios_no_mag = 0
        except cdas.NoDataError:
            mag = {}
            helios_no_mag = 1
        pla = cdas.get_data(
            'sp_phys', 'HELIOS2_40SEC_MAG-PLASMA', t_start, t_end,
            ['B_R', 'B_T', 'B_N',
             'Np', 'Tp', 'Vp',
             'Vp_R', 'Vp_T', 'Vp_N'])
        pos =  cdas.get_data(
            'sp_phys', 'HELIOS2_40SEC_MAG-PLASMA', t_start, t_end,
            ['R_Helio', 'clat', 'HGIlong'])

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VP_R']
        Vy = pla['VP_T']
        Vz = pla['VP_N']
        V = pla['V_P']
        Np = pla['N_P']
        Tp = pla['T_P']

        # Additional magnetic field data from plasma datasets:
        # These will be used later if the primary mag data is faulty
        t_mag_add = pla['EPOCH']
        Bx_add = pla['B_R']
        By_add = pla['B_T']
        Bz_add = pla['B_N']
        B_add = np.sqrt(Bx_add**2 + By_add**2 + Bz_add**2)

        if helios_no_mag == 0:
            t_mag = mag['EPOCH']
            Bx = -1 * mag['BX_(SSE)']
            By = -1 * mag['BY_(SSE)']
            Bz = mag['BZ_(SSE)']
            B = mag['B']
        else:
            t_mag = t_mag_add
            Bx = Bx_add
            By = By_add
            Bz = Bz_add
            B = B_add

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['R_HELIO']
        pos_Y = pos['CARRINGTON/HGI_LATITUDE']
        pos_Z = pos['HGI_LONGITUDE']
    elif sc == 6:  # Ulysses
        mag = cdas.get_data(
            'sp_phys', 'UY_1SEC_VHM', t_start, t_end, ['B_MAG', 'B_RTN'])
        pla = cdas.get_data(
            'sp_phys', 'UY_M0_BAI', t_start, t_end,
            ['Density', 'Temperature', 'Velocity'])
        pos = cdas.get_data(
            'sp_phys', 'UY_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end,
            ['heliocentricDistance', 'heliographicLatitude',
             'heliographicLongitude'])
        pos_vec = [pos['ULYSSES_DIST'], pos['HGI_LAT'], pos['HGI_LONG']]

        # Magnetic field data
        t_mag = mag['EPOCH_TIME']
        Bx = mag['R_COMPONENT']
        By = mag['T_COMPONENT']
        Bz = mag['N_COMPONENT']
        B = mag['MAGNETIC_FIELD_MAGNITUDE']

        # Plasma data
        t_pla = pla['TIME']
        Vx = pla['VEL_R']
        Vy = pla['VEL_T']
        Vz = pla['VEL_N']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['PROTON']

        # Calculate the mean of the minimum and maximum value
        # temperature datasets
        Tmax = pla['T-LARGE']
        Tmin = pla['T-SMALL']
        Tp = (Tmax + Tmin)/2

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['ULYSSES_DIST']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']
    elif sc == 7:  # Cluster 3
        mag = cdas.get_data(
            'sp_phys', 'C3_CP_FGM_SPIN', t_start, t_end,
            ['B_mag__C3_CP_FGM_SPIN', 'B_vec_xyz_gse__C3_CP_FGM_SPIN'])
        pla = cdas.get_data(
            'sp_phys', 'C3_PP_CIS', t_start, t_end,
            ['Status__C3_PP_CIS', 'N_HIA__C3_PP_CIS',
             'V_HIA_xyz_gse__C3_PP_CIS', 'T_HIA_par__C3_PP_CIS',
             'T_HIA_perp__C3_PP_CIS'])
        pos = cdas.get_data(
            'sp_phys', 'C3_CP_FGM_SPIN', t_start, t_end,
            ['sc_pos_xyz_gse__C3_CP_FGM_SPIN'])

        # Magnetic field data
        t_mag = mag['UT']
        Bx = mag['BX_GSE']
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        B = mag['B']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VX_HIA_GSE']
        Vy = pla['VY_HIA_GSE']
        Vz = pla['VZ_HIA_GSE']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['N(HIA)']
        Tp_par = pla['T(HIA)_PAR']
        Tp_perp = pla['T(HIA)_PERP']
        status = pla['STATUS[1]']  # This is used to check if the data was okay

        # Total proton temperature assuming an isotropic velocity
        # distribution. The temperature is converted from MK to K
        Tp = ((2/3)*Tp_perp + (1/3)*Tp_par)*1e6

        # Position data
        t_pos = pos['UT']
        pos_X = pos['C3_X_GSE']
        pos_Y = pos['C3_Y_GSE']
        pos_Z = pos['C3_Z_GSE']
    elif sc == 8:  # Cluster 1
        mag = cdas.get_data(
            'sp_phys', 'C1_CP_FGM_SPIN', t_start, t_end,
            ['B_mag__C1_CP_FGM_SPIN', 'B_vec_xyz_gse__C1_CP_FGM_SPIN'])
        pla = cdas.get_data(
            'sp_phys', 'C1_PP_CIS', t_start, t_end,
            ['Status__C1_PP_CIS', 'N_HIA__C1_PP_CIS',
             'V_HIA_xyz_gse__C1_PP_CIS', 'T_HIA_par__C1_PP_CIS',
             'T_HIA_perp__C1_PP_CIS'])
        pos = cdas.get_data(
            'sp_phys', 'C1_CP_FGM_SPIN', t_start, t_end,
            ['sc_pos_xyz_gse__C1_CP_FGM_SPIN'])

        # Magnetic field data
        t_mag = mag['UT']
        Bx = mag['BX_GSE']
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        B = mag['B']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VX_HIA_GSE']
        Vy = pla['VY_HIA_GSE']
        Vz = pla['VZ_HIA_GSE']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['N(HIA)']
        Tp_par = pla['T(HIA)_PAR']
        Tp_perp = pla['T(HIA)_PERP']
        status = pla['STATUS[1]']  # This is used to check if the data was okay

        # Total proton temperature assuming an isotropic velocity
        # distribution. The temperature is converted from MK to K
        Tp = ((2/3)*Tp_perp + (1/3)*Tp_par)*1e6

        # Position data
        t_pos = pos['UT']
        pos_X = pos['C1_X_GSE']
        pos_Y = pos['C1_Y_GSE']
        pos_Z = pos['C1_Z_GSE']
    if sc == 9:  # Cluster 4
        mag = cdas.get_data(
            'sp_phys', 'C4_CP_FGM_SPIN', t_start, t_end,
            ['B_mag__C4_CP_FGM_SPIN', 'B_vec_xyz_gse__C4_CP_FGM_SPIN'])
        pla = cdas.get_data(
            'sp_phys', 'C4_PP_CIS', t_start, t_end,
            ['Status__C4_PP_CIS', 'N_p__C4_PP_CIS',
             'V_p_xyz_gse__C4_PP_CIS', 'T_p_par__C4_PP_CIS',
             'T_p_perp__C4_PP_CIS'])
        pos = cdas.get_data(
            'sp_phys', 'C4_CP_FGM_SPIN', t_start, t_end,
            ['sc_pos_xyz_gse__C4_CP_FGM_SPIN'])

        # Magnetic field data
        t_mag = mag['UT']
        Bx = mag['BX_GSE']
        By = mag['BY_GSE']
        Bz = mag['BZ_GSE']
        B = mag['B']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VX_P_GSE']
        Vy = pla['VY_P_GSE']
        Vz = pla['VZ_P_GSE']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['N(P)']
        Tp_par = pla['T(P)_PAR']
        Tp_perp = pla['T(P)_PERP']
        status = pla['STATUS[1]']  # This is used to check if the data was okay

        # Total proton temperature assuming an isotropic velocity
        # distribution. The temperature is converted from MK to K
        Tp = ((2/3)*Tp_perp + (1/3)*Tp_par)*1e6

        # Position data
        t_pos = pos['UT']
        pos_X = pos['C4_X_GSE']
        pos_Y = pos['C4_Y_GSE']
        pos_Z = pos['C4_Z_GSE']
    if sc == 10:  # OMNI
        mag = cdas.get_data(
            'sp_phys', 'OMNI_HRO_1MIN', t_start, t_end,
            ['F', 'BX_GSE', 'BY_GSE', 'BZ_GSE'])
        pla = cdas.get_data(
            'sp_phys', 'OMNI_HRO_1MIN', t_start, t_end,
            ['flow_speed', 'Vx', 'Vy', 'Vz', 'proton_density', 'T'])
        pos = cdas.get_data(
            'sp_phys', 'OMNI_HRO_1MIN', t_start, t_end,
            ['BSN_x', 'BSN_y', 'BSN_z'])
        pos_vec = [
            pos['X_(BSN),_GSE'], pos['Y_(BSN),_GSE'], pos['Z_(BSN),_GSE']]
        t_pos = pos['EPOCH_TIME']

        # Magnetic field data
        t_mag = mag['EPOCH_TIME']
        Bx = mag['BX,_GSE']
        By = mag['BY,_GSE']
        Bz = mag['BZ,_GSE']
        B = mag['MAG_AVG_B-VECTOR']

        # Plasma data
        t_pla = pla['EPOCH_TIME']
        Vx = pla['VX_VELOCITY,_GSE']
        Vy = pla['VY_VELOCITY,_GSE']
        Vz = pla['VZ_VELOCITY,_GSE']
        V = pla['FLOW_SPEED,_GSE']
        Np = pla['PROTON_DENSITY']
        Tp = pla['TEMPERATURE']

        # Position data
        t_pos = pos['EPOCH_TIME']
        pos_X = pos['X_(BSN),_GSE']
        pos_Y = pos['Y_(BSN),_GSE']
        pos_Z = pos['Z_(BSN),_GSE']
    if sc == 11:  # Voyager 1
        mag = cdas.get_data(
            'sp_phys', 'VOYAGER1_2S_MAG', t_start, t_end,
            ['F1', 'B1', 'B2', 'B3'])
        pla = cdas.get_data(
            'sp_phys', 'VOYAGER1_PLS_HIRES_PLASMA_DATA', t_start, t_end,
            ['V', 'V_rtn', 'dens', 'V_thermal'])
        pos = cdas.get_data(
            'sp_phys', 'VOYAGER1_2S_MAG', t_start, t_end,
            ['scDistance', 'scLon', 'scLat'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BR_(B1)']
        By = mag['BT_(B2)']
        Bz = mag['BN_(B3)']
        B = mag['B-MAGNITUDE_(F1)']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VR']
        Vy = pla['VT']
        Vz = pla['VN']
        V = pla['VP']
        Np = pla['PROTON_DENSITY']
        V_th = pla['VP_THERMAL']

        # Convert thermal speed to temperature
        Tp = thermal_speed_to_temperature(V_th)

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['V1_DIST']
        pos_Y = np.rad2deg(pos['V1_LAT_IHG'])
        pos_Z = np.rad2deg(pos['V1_LONG_IHG'])
    if sc == 12:  # Voyager 2
        mag = cdas.get_data(
            'sp_phys', 'VOYAGER2_2S_MAG', t_start, t_end,
            ['F1', 'B1', 'B2', 'B3'])
        pla = cdas.get_data(
            'sp_phys', 'VOYAGER2_PLS_HIRES_PLASMA_DATA', t_start, t_end,
            ['V', 'V_rtn', 'dens', 'V_thermal'])
        pos = cdas.get_data(
            'sp_phys', 'VOYAGER2_2S_MAG', t_start, t_end,
            ['scDistance', 'scLon', 'scLat'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BR_(B1)']
        By = mag['BT_(B2)']
        Bz = mag['BN_(B3)']
        B = mag['B-MAGNITUDE_(F1)']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VR']
        Vy = pla['VT']
        Vz = pla['VN']
        V = pla['VP']
        Np = pla['PROTON_DENSITY']
        V_th = pla['VP_THERMAL']

        # Convert thermal speed to temperature
        Tp = thermal_speed_to_temperature(V_th)

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['V1_DIST']
        pos_Y = np.rad2deg(pos['V1_LAT_IHG'])
        pos_Z = np.rad2deg(pos['V1_LONG_IHG'])
    if sc == 13:  # DSCOVR
        mag = cdas.get_data(
            'sp_phys', 'DSCOVR_H0_MAG', t_start, t_end, ['B1F1', 'B1GSE'])
        pla = cdas.get_data(
            'sp_phys', 'DSCOVR_H1_FC', t_start, t_end,
            ['V_GSE', 'Np', 'THERMAL_TEMP'])
        pos = cdas.get_data(
            'sp_phys', 'DSCOVR_ORBIT_PRE', t_start, t_end, ['GSE_POS'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['BX_(GSE)']
        By = mag['BY_(GSE)']
        Bz = mag['BZ_(GSE)']
        B = mag['B']

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VX_(GSE)']
        Vy = pla['VY_(GSE)']
        Vz = pla['VZ_(GSE)']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['ION_N']
        Tp = pla['TEMPERATURE']

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['X_GSE']
        pos_Y = pos['Y_GSE']
        pos_Z = pos['Z_GSE']
    if sc == 14:  # PSP
        mag = cdas.get_data(
            'sp_phys', 'PSP_FLD_L2_MAG_RTN_1MIN', t_start, t_end,
            ['psp_fld_l2_mag_RTN_1min'])
        pla = cdas.get_data(
            'sp_phys', 'PSP_SWP_SPC_L3I', t_start, t_end,
            ['np_moment_gd', 'vp_moment_RTN_gd', 'wp_moment'])
        pos_HCI = cdas.get_data(
            'sp_phys', 'PSP_SWP_SPC_L3I', t_start, t_end, ['sc_pos_HCI'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['B_R']
        By = mag['B_T']
        Bz = mag['B_N']
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VP_MOMENT_R']
        Vy = pla['VP_MOMENT_T']
        Vz = pla['VP_MOMENT_N']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['NP_MOMENT']
        V_th = pla['WP_MOMENT']

        # Convert thermal speed to temperature.
        Tp = thermal_speed_to_temperature(V_th)

        # Position data in heliocentric inertial coordinates (HCI)
        t_pos = pos_HCI['EPOCH']
        pos_X = pos_HCI['X_HCI']
        pos_Y = pos_HCI['Y_HCI']
        pos_Z = pos_HCI['Z_HCI']

        # Convert HCI to (rad, lat, long) format, where rad is given in
        # astronomical units and lat and long in degrees
        rad_dist = np.sqrt(pos_X**2 + pos_Y**2 + pos_Z**2)*(1e3/constants.au)
        lat = np.rad2deg(np.arctan(pos_Z/np.sqrt(pos_X**2 + pos_Y**2)))
        long = np.mod(np.rad2deg(np.arctan2(pos_Y, pos_X)), 360)

        # Replace position in HCI with position in (rad, lat, long)
        pos_X = rad_dist
        pos_Y = lat
        pos_Z = long
    elif sc == 15:  # SolO
        mag = cdas.get_data(
            'sp_phys', 'SOLO_L2_MAG-RTN-NORMAL', t_start, t_end, ['B_RTN'])
        pla = cdas.get_data(
            'sp_phys', 'SOLO_L2_SWA-PAS-GRND-MOM', t_start, t_end,
            ['N', 'V_RTN', 'T'])
        pos = cdas.get_data(
            'sp_phys', 'SOLO_COHO1HR_MERGED_MAG_PLASMA', t_start, t_end,
            ['radialDistance', 'heliographicLatitude',
             'heliographicLongitude'])

        # Magnetic field data
        t_mag = mag['EPOCH']
        Bx = mag['B_R']
        By = mag['B_T']
        Bz = mag['B_N']
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

        # Plasma data
        t_pla = pla['EPOCH']
        Vx = pla['VR_RTN']
        Vy = pla['VT_RTN']
        Vz = pla['VN_RTN']
        V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        Np = pla['DENSITY']
        Tp = pla['TEMPERATURE']

        # Convert temperature from eV to K
        Tp = Tp*(constants.eV/constants.Boltzmann)

        # Position data
        t_pos = pos['EPOCH']
        pos_X = pos['RADIAL_DISTANCE']
        pos_Y = pos['HGI_LAT']
        pos_Z = pos['HGI_LONG']

    # ------------------------------------------------------------------
    # Creating pandas DataFrames of the data products
    # ------------------------------------------------------------------

    # If the spacecraft is not one of the Cluster spacecraft, create
    # a status array full of NaNs
    if sc not in [7, 8, 9]:
        status = np.full(len(t_pla), np.nan)

    # Convert all time vectors from datetime format to Unix time,
    # i.e., seconds since January 1, 1970
    t_mag = [dt.timestamp() for dt in t_mag]
    t_mag_add = [dt.timestamp() for dt in t_mag_add]
    t_pla = [dt.timestamp() for dt in t_pla]
    t_pos = [dt.timestamp() for dt in t_pos]

    # Create DataFrames for magnetic field, plasma, and position data
    mag_dataframe = pd.DataFrame({
        'EPOCH': t_mag,
        'Bx': Bx,
        'By': By,
        'Bz': Bz,
        'B': B})
    pla_dataframe = pd.DataFrame({
        'EPOCH': t_pla,
        'Vx': Vx,
        'Vy': Vy,
        'Vz': Vz,
        'V': V,
        'Np': Np,
        'Tp': Tp,
        'status': status})
    pos_dataframe = pd.DataFrame({
        'EPOCH': t_pos,
        'pos_X': pos_X,
        'pos_Y': pos_Y,
        'pos_Z': pos_Z})

    # Additional DataFrame for Helios
    if sc in [4, 5]:
        add_dataframe = pd.DataFrame({
            'EPOCH': t_mag_add,
            'Bx': Bx_add,
            'By': By_add,
            'Bz': Bz_add,
            'B': B_add})
    else:
        add_dataframe = pd.DataFrame()

    # ------------------------------------------------------------------
    # Cleaning the data (i.e., removing bad values)
    # ------------------------------------------------------------------

    # Magnetic field components
    mag_dataframe['Bx'] = mag_dataframe['Bx'].mask(
        np.abs(mag_dataframe['Bx']) > 1500)
    mag_dataframe['By'] = mag_dataframe['By'].mask(
        np.abs(mag_dataframe['By']) > 1500)
    mag_dataframe['Bz'] = mag_dataframe['Bz'].mask(
        np.abs(mag_dataframe['Bz']) > 1500)

    # Magnetic field magnitude
    mag_dataframe['B'] = mag_dataframe['B'].mask(
        (mag_dataframe['B'] > 150) | (mag_dataframe['B'] <= 0))

    # Solar wind velocity components
    pla_dataframe['Vx'] = pla_dataframe['Vx'].mask(
        np.abs(pla_dataframe['Vx']) > 1500)
    pla_dataframe['Vy'] = pla_dataframe['Vy'].mask(
        np.abs(pla_dataframe['Vy']) > 1500)
    pla_dataframe['Vz'] = pla_dataframe['Vz'].mask(
        np.abs(pla_dataframe['Vz']) > 1500)

    # Solar wind bulk speed
    pla_dataframe['V'] = pla_dataframe['V'].mask(
        (pla_dataframe['V'] > 1500) | (pla_dataframe['V'] < 0))

    # Plasma density
    pla_dataframe['Np'] = pla_dataframe['Np'].mask(
        (pla_dataframe['Np'] > 150) | (pla_dataframe['Np'] < 0))

    # Plasma temperature
    pla_dataframe['Tp'] = pla_dataframe['Tp'].mask(
        (pla_dataframe['Tp'] > 0.9e7) | (pla_dataframe['Tp'] < 0))

    # Position
    pos_dataframe['pos_X'] = pos_dataframe['pos_X'].mask(
        np.abs(pos_dataframe['pos_X']) > 1e9)
    pos_dataframe['pos_Y'] = pos_dataframe['pos_Y'].mask(
        np.abs(pos_dataframe['pos_Y']) > 1e9)
    pos_dataframe['pos_Z'] = pos_dataframe['pos_Z'].mask(
        np.abs(pos_dataframe['pos_Z']) > 1e9)

    # For the Cluster plasma data, delete values where the instrument
    # status was wrong (5 or higher)
    if sc in [7, 8, 9]:
        status = pla_dataframe['status']
        faulty_values = status > 5
        pla_dataframe['V'] = pla_dataframe['V'].mask(faulty_values)
        pla_dataframe['Vx'] = pla_dataframe['Vx'].mask(faulty_values)
        pla_dataframe['Vy'] = pla_dataframe['Vy'].mask(faulty_values)
        pla_dataframe['Vz'] = pla_dataframe['Vz'].mask(faulty_values)
        pla_dataframe['Np'] = pla_dataframe['Np'].mask(faulty_values)
        pla_dataframe['Tp'] = pla_dataframe['Tp'].mask(faulty_values)
        pla_dataframe = pla_dataframe.drop(columns=['status'])

    # Clean the additional Helios DataFrame
    if sc in [4, 5]:
        condition = (add_dataframe['B'] > 150) | (add_dataframe['B'] <= 0)
        add_dataframe['B'] = add_dataframe['B'].mask(condition)
        add_dataframe['Bx'] = add_dataframe['Bx'].mask(condition)
        add_dataframe['By'] = add_dataframe['By'].mask(condition)
        add_dataframe['Bz'] = add_dataframe['Bz'].mask(condition)

        condition = ((add_dataframe['Bx'] > 1500)
                     | (add_dataframe['By'] > 1500)
                     | (add_dataframe['By'] > 1500))
        add_dataframe['B'] = add_dataframe['B'].mask(condition)
        add_dataframe['Bx'] = add_dataframe['Bx'].mask(condition)
        add_dataframe['By'] = add_dataframe['By'].mask(condition)
        add_dataframe['Bz'] = add_dataframe['Bz'].mask(condition)

    # ------------------------------------------------------------------
    # If filtering was chosen by the user, use the median filter to
    # filter out data spikes
    # ------------------------------------------------------------------

    if isinstance(filter_options, int):
        filter_options = [filter_options]

    if filter_options[0] != 0:
        if filter_options[0] == 1:  # Default settings
            median_window_size = 5
            tols = [0.75, 1.5, 0.2]
        elif len(filter_options) == 4:  # User-defined settings
            median_window_size = filter_options[0]
            tols = filter_options[1:4]
        else:  # Incorrect input
            raise ValueError(
                'Incorrect input for the spike filter. '
                'Must be either 0, 1, or a 4-element float array')

        # Take a running median of the Np, Tp and V Series and remove
        # values that differ from the median Series too much. The
        # allowed difference is set by the corresponding multiplier in
        # the filter_options array. The options in the filter_options
        # array are [width, Np, Tp, V].
        for var in ['Np', 'Tp', 'V']:
            pla_dataframe[var + '_median'] = pla_dataframe[var].rolling(
                window=int(median_window_size), center=True).median()
            
            # Calculate the absolute difference from the median
            diff = np.abs(pla_dataframe[var] - pla_dataframe[var + '_median'])

            # Calculate the tolerance limit based on the multiplier
            tol_limit = (tols[['Np', 'Tp', 'V'].index(var)]
                         * pla_dataframe[var + '_median'])

            # Replace values exceeding the tolerance with NaN
            pla_dataframe.loc[diff > tol_limit, var] = np.nan

        # Find indices where V contains NaN values
        missing_velocity = pla_dataframe['V'].isna()

        # Set corresponding elements in Vx, Vy, and Vz to NaN
        pla_dataframe.loc[missing_velocity, ['Vx', 'Vy', 'Vz']] = np.nan


    # ------------------------------------------------------------------
    # If the given shock time is only preliminary, a better estimate is
    # determined using magnetic field data
    # ------------------------------------------------------------------

    # THIS FEATURE HAS BEEN TURNED OFF AS SHOCK TIME IS FINE-TUNED BY
    # THE USER

    # # Preliminary (t_pre = 1) / not preliminary (t_pre = 0)
    # if t_pre == 1:
    #     t_shock = check_shock_time(
    #        mag_dataframe['EPOCH'], mag_dataframe['B'], shock_timestamp)
        
    #     # Additional shock time estimate based on additional Helios
    #     # magnetic field data
    #     if (sc == 4) or (sc == 5) and (helios_no_mag == 0):
    #         t_shock_new_add = check_shock_time(
    #             add_dataframe['EPOCH'], add_dataframe['B'], shock_timestamp)
    #     else:
    #         t_shock_new_add = np.nan
        
    #     t_shock_new = t_shock
    # else:
    #     # if the shock time is not preliminary, the new time is the same
    #     t_shock_new = shock_timestamp
    #     t_shock_new_add = shock_timestamp

    # ------------------------------------------------------------------
    # Collecting the position data around the time of the shock
    # ------------------------------------------------------------------

    # Find the index of the closest time point to the shock time
    idx = np.argmin(np.abs(pos_dataframe['EPOCH'] - shock_timestamp))

    # Extract the position vector at the closest time point
    sc_pos = pos_dataframe.loc[idx, ['pos_X', 'pos_Y', 'pos_Z']]

    # Additional position vector based on additional Helios data
    if (sc == 4) or (sc == 5) and (helios_no_mag == 0):
        idx = np.argmin(np.abs(add_dataframe['EPOCH'] - shock_timestamp))
        sc_pos_add = pos_dataframe.loc[idx, ['pos_X', 'pos_Y', 'pos_Z']]

    # Positions of ACE, Cluster, and DSCOVR spacecraft are in km,
    # rescale them to be expressed in Earth radii (R_E = 6378.14 km)
    scaling_factor = 6378.14
    if sc in [0, 7, 8, 9, 13]:
        sc_pos /= scaling_factor

    # ------------------------------------------------------------------
    # Determining the validity of the spacecraft's position vector at
    # the time of shock (POSITION DATA IS RARELY INVALID. DO NOT CHANGE
    # THIS SECTION UNLESS AN ADDITIONAL DATA SOURCE TO REPLACE THE
    # INVALID POSITION DATA CAN BE DETERMINED AND UTILISED)
    # ------------------------------------------------------------------
    
    # NOT IMPLEMENTED IN THIS PYTHON VERSION AT THIS TIME

    # # Check if the position vector is valid (pos_ch = 1 for a valid
    # # vector, and pos_ch = 0 for an invalid vector)
    # pos_ch = 1
    # faulty_values = np.where(np.isfinite(sc_pos))
    # if len(faulty_values[0]) < 3:
    #     pos_ch = 0

    # # If the position vector is not valid, other data sources are used
    # # (if possible)

    # # ACE has one additional source
    # if (sc == 0) and (pos_ch == 0):
    #    pos = cdas.get_data(
    #        'sp_phys', 'AC_H0_SWE', t_start, t_end, ['SC_pos_GSE'])
    
    #    pos_vec = np.array(
    #        [pos['ACE_X-GSE'], pos['ACE_Y-GSE'], pos['ACE_Z-GSE']])
    #    t_pos = np.array([dt.timestamp() for dt in [pos['EPOCH']]])

    #    idx = np.argmin(np.abs(shock_timestamp - t_pos))
    #    sc_pos = pos_vec[:, idx]/scaling_factor

    # # Wind has two additional sources
    # if (sc == 1) and (pos_ch == 0):
    #     # The first source
    #     pos = cdas.get_data(
    #         'sp_phys', 'WI_K0_SWE', t_start, t_end, ['SC_pos_gse'])
        
    #     pos_vec = np.array(
    #         [pos['WI_X_(GSE)'], pos['WI_Y_(GSE)'], pos['WI_Z_(GSE)']])
    #     t_pos = np.array([dt.timestamp() for dt in [pos['EPOCH']]])
        
    #     idx = np.argmin(np.abs(shock_timestamp - t_pos))
    #     sc_pos = pos_vec[:, idx]

    #     # Validity check
    #     pos_ch = 1
    #     faulty_values = np.where(np.isfinite(sc_pos))
    #     if len(faulty_values[0]) < 3:
    #         pos_ch = 0

    #     # The second source (used only if the first source does not
    #     # give a valid result)
    #     if pos_ch == 0:
    #         # The source for the second additional position data depends
    #         # on the time (before or after 1.7.1997 23:50)
    #         wind_limit = datetime(1997, 7, 1, 23, 50).timestamp()
    #         if shock_timestamp >= wind_limit:
    #             pos_title = 'WI_OR_PRE'
    #         else:
    #             pos_title = 'WI_OR_DEF'
            
    #         pos = cdas.get_data(
    #             'sp_phys', pos_title, t_start, t_end, ['GSE_POS'])
            
    #         pos_vec = np.array(
    #             [pos['GSE_X'], pos['GSE_Y'], pos['GSE_Z']])
    #         t_pos = np.array([dt.timestamp() for dt in [pos['EPOCH']]])
            
    #         idx = np.argmin(np.abs(shock_timestamp - t_pos))
    #         sc_pos = pos_vec[:, idx]

    #     sc_pos = sc_pos/scaling_factor

    # ------------------------------------------------------------------
    # The final output
    # ------------------------------------------------------------------

    # Additional Helios output
    if (sc == 4) or (sc == 5) and (helios_no_mag == 0):
        output_add = [add_dataframe, shock_timestamp, sc_pos_add]
    else:
        output_add = [add_dataframe, shock_timestamp, sc_pos]

    # Information of the measurement radii is given as an output (this
    # only applies to the Wind spacecraft)
    if sc == 1:
        pla_bin_rads = bin_rad_wind_pla
    else:
        pla_bin_rads = 0

    return (mag_dataframe, pla_dataframe, sc_pos, output_add,
            shock_timestamp, pla_bin_rads)
