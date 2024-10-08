# ----------------------------------------------------------------------
# The Shock Analysis Toolkit (IPSKIT) enables the user to analyse fast
# interplanetary shock waves detected in spacecraft data. The toolkit
# provides all the necessary automatised parts for the analysis. It
# produces an output data file as well as PS and PNG plots of the
# analysed shocks. The toolkit is used to update the Database of
# Heliospheric Shocks Waves (IPShocks; https://ipshocks.helsinki.fi)
# maintained at the University of Helsinki. The running of the shock
# analysis program is explained in the associated documentation file,
# while the basics of the analysis methods used by the toolkit are
# described at https://ipshocks.helsinki.fi/documentation.
#
# This toolkit was originally developed by Erkka Lumme in 2017, and it
# was written in IDL. It was translated to Python by Timo Mäkelä in
# 2024. Further edits have been made by Juska Soljento. During
# translation the Parker Solar Probe (PSP) and Solar Orbiter (SolO)
# spacecraft were added, and the code was simplified to better suit the
# current workflow where shocks are found using the automated machine
# learning algorithm IPSVM (https://bitbucket.org/raysofspace/ipsvm).
# ----------------------------------------------------------------------

from datetime import datetime, timedelta
import os
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from download_and_process_data import download_and_process_data
from error_analysis import error_analysis


# ----------------------------------------------------------------------
# FUNCTIONS AND PROCEDURES REQUIRED BY THE SHOCK ANALYSIS PROGRAM
# ----------------------------------------------------------------------

def analysis_interval_check(data, var_names, t_up, t_down):
    """Check that there is enough data to proceed with the analysis.

    This function checks that there are at least three data points in
    both the upstream and downstream intervals of the dataframe.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame.
    var_names : array_like
        Array of variables to loop through.
    t_up : array_like
        Start and end time of the upstream interval.
    t_down : array_like
        Start and end time of the downstream interval.

    Returns
    -------
    too_few_datapoints : bool
        If True, there are less than three data points, if False, there
        are at least three data points.
    """

    # Initialize the returned Boolean
    too_few_datapoints = False

    # Loop through the given variables
    for var in var_names:
        # Choose the data in the upstream and downstream intervals
        data_up = data[
            (data['EPOCH'] >= t_up[0]) & (data['EPOCH'] <= t_up[1])][var]
        data_down = data[
            (data['EPOCH'] >= t_down[0]) & (data['EPOCH'] <= t_down[1])][var]

        # Count the datapoints in the intervals
        points_up = data_up.notna().sum()
        points_down = data_down.notna().sum()

        # Return True if either the upstream or the downstream contain
        # less than three data points
        if points_up < 3 or points_down < 3:
            too_few_datapoints = True
            break

    return too_few_datapoints


def return_resolution(var_name, sc, helios_flag):
    """Return the data resolution for a given spacecraft.

    Parameters
    ----------
    var_name : str
        Name of the variable.
    sc : int
        The spacecraft ID number.
    helios_flag : int
        Helios data flag, 0 for 6 sec magnetic field data and 1 for
        40.5 sec magnetic field data.

    Returns
    -------
    res : float
        Resolution of the data product (in seconds).
    """

    mag_res = None
    pla_res = None

    if sc == 0:
        mag_res = 16.
        pla_res = 64.
    elif sc == 1:
        mag_res = 3.
        pla_res = 90.
    elif sc in [2, 3]:
        mag_res = 0.125
        pla_res = 60.
    elif (sc in [4, 5]) and (helios_flag == 0):
        mag_res = 6.
        pla_res = 40.5
    elif (sc in [4, 5]) and (helios_flag == 1):
        mag_res = 40.5
        pla_res = 40.5
    elif sc == 6:
        pla_res = 240.
        mag_res = 1.
    elif sc in [7, 8, 9]:
        mag_res = 4.
        pla_res = 4.
    elif sc == 10:
        mag_res = 60.
        pla_res = 60.
    elif sc in [11, 12]:
        mag_res = 1.92
        pla_res = 12.
    elif sc == 13:
        mag_res = 1.
        pla_res = 60.
    elif sc == 14:
        mag_res = 60.
        pla_res = 0.87
    elif sc == 15:
        mag_res = 0.126
        pla_res = 4.0

    # Return the resolution corresponding to the input variable
    res = mag_res if var_name.startswith('B') else pla_res

    return res


def mean_data(data, var_names, interval):
    """Calculate the mean and standard deviation of a data product.

    This function calculates the mean and standard deviation for the
    given variables over a given time interval and returns them as
    dictionaries.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data for which the mean and standard
        deviation are calculated.
    var_names : array_like
        Array of the names of the variables.
    interval : array_like
        Start and end of the time interval.

    Returns
    -------
    mean_vals : dict
        Dictionary of the means for each data product.
    std_vals : dict
        Dictionary of the standard deviations for each data product.
    """

    t_start, t_end = interval
    mean_vals = {}
    std_vals = {}

    for var in var_names:
        if var in data:
            data_interval = data[
                (data['EPOCH'] >= t_start) & (data['EPOCH'] <= t_end)][var]
            mean_vals[var] = data_interval.mean(skipna=True)
            std_vals[var] = data_interval.std(skipna=True)

            # If standard deviation is not finite, set it to 0
            if not np.isfinite(std_vals[var]):
                std_vals[var] = 0
        else:
            mean_vals[var] = np.nan
            std_vals[var] = 0

    return mean_vals, std_vals


def calculate_mean_resolution(data, var_names, t_up, t_down, sc, helios_flag):
    """Calculate the mean data resolution over a time interval.

    This function calculates the mean data resolution for the given
    variables over a given time interval and returns the resulting
    maximum resolution. If there is insufficient data to perform the
    calculation, the function returns False.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data for which the mean resolution is
        calculated.
    var_names : array_like
        Array of variables to loop through.
    t_up : array_like
        Start and end time of the upstream interval.
    t_down : array_like
        Start and end time of the downstream interval.
    sc : int
        The spacecraft ID number.
    helios_flag : int
        Helios data flag, 0 for 6 sec magnetic field data and 1 for
        40.5 sec magnetic field data.

    Returns
    -------
    avg_res_max : float or bool
        Mean data resolution as a float. Returns False if there is
        insufficient data to perform the calculation.
    """

    interval_start = min(t_up[0], t_up[1], t_down[0], t_down[1])
    interval_end = max(t_up[0], t_up[1], t_down[0], t_down[1])
    avg_res_max = 0

    for var in var_names:
        if var in data:
            data_interval = data[
                (data['EPOCH'] >= interval_start)
                & (data['EPOCH'] <= interval_end)
                & data[var].notna()][var]

            # Count non-null values after applying the mask
            good_points = data_interval.count()

            if good_points == 0:
                print("Not enough data for variable: " + str(var))
                return False

            if good_points < 6 and sc in [4, 5, 6]:
                return False

            res = return_resolution(var, sc, helios_flag)
            avg_res = res
            pnts_reg = (interval_end - interval_start) // res

            if pnts_reg > good_points:
                avg_res = (interval_end - interval_start) / good_points

            if avg_res > avg_res_max:
                avg_res_max = avg_res

    return avg_res_max


def resample(data, t_original, t_lower_res, interval, sc, bin_rad=None):
    """Resample input data to a lower resolution.

    Parameters
    ----------
    data : array_like
        Data that is resampled to a lower resolution.
    t_original : array_like
        Time coordinates of the original data.
    t_lower_res : array_like
        Lower resolution array of times. The data is resampled to match
        this resolution.
    interval : array_like
        Interval over which resampling is performed.
    sc : int
        Spacecraft ID number. See documentation for details.
    bin_rad : array_like
        A radius around each data point. For example ACE's plasma data
        resolution is 64 s, which defines a bin radius of 32 s on either
        side of each data point. Wind has non-constant plasma data
        resolution so this will be an array, for the other spacecraft
        this is a float.

    Returns
    -------
    resampled_data : array_like
        Array of resampled data.
    """

    # Count the number of time ticks in the lower resolution series
    # within the interval
    interval_mask = np.where(
        (t_lower_res >= interval[0]) & (t_lower_res <= interval[1]))[0]
    interval_times = t_lower_res[interval_mask]
    n_interval_ticks = len(interval_times)

    # Determine the minimum resolution over all the considered plasma
    # data points
    ind_pla = np.concatenate(
        ([interval_mask[0] - 1], interval_mask, [interval_mask[-1] + 1]))
    diffs = np.abs(np.diff(t_lower_res[ind_pla]))
    min_res = np.min(diffs[1:-1])

    # For Wind there may be a different radius for each plasma data
    # point, i.e., the plasma data resolution is not necessarily
    # constant
    if sc == 1:
        bin_rad_up = bin_rad[interval_mask]
    else:
        bin_rad_up = 0

    # Determine the measurement bin radii
    t_bins = np.zeros((n_interval_ticks, 2))
    if bin_rad is not None:
        for i in range(n_interval_ticks):
            if sc == 1:
                rad = np.abs(bin_rad_up[i])
            else:
                rad = np.abs(bin_rad)

            local_min_res = np.min(diffs[i + 1:i + 2])

            if rad > (local_min_res / 2):
                rad = local_min_res / 2

            # Special case for the Voyager spacecraft
            if sc in [11, 12] and (local_min_res / 2) > rad:
                rad = local_min_res / 2

            t_bins = np.array(
                [[time - rad, time + rad] for time in interval_times])
    else:
        for i in range(n_interval_ticks):
            t_bins = np.array(
                [[time - min_res / 2, time + min_res / 2]
                 for time in interval_times])

    # Initialize an array where the averaged values will be added
    resampled_data = np.zeros(n_interval_ticks)

    # Loop through the bins and average the data
    for j in range(n_interval_ticks):
        # Find indices of t_original which are within the current bin
        # range
        indices = np.where(
            (t_original > t_bins[j, 0]) & (t_original <= t_bins[j, 1]))[0]

        if len(indices) > 0:
            # Average values in data corresponding to these indices
            resampled_data[j] = np.nanmean(data[indices])
        else:
            # If no indices fall within the bin, assign NaN
            resampled_data[j] = np.nan

    return resampled_data


# Global variables to mimic IDL's COMMON block
normal_eq_input = {}


def solve_normal_eq(vec_1, vec_2):
    """Solve the normal equation to find the shock normal.

    This function takes two vectors (which are different combinations of
    the magnetic field and plasma velocity on either side of the shock),
    and uses scipy.optimize.fsolve to determine the shock normal vector.

    Parameters
    ----------
    vec_1 : array_like
        First vector input.
    vec_2 : array_like
        Second vector input.

    Returns
    -------
    normal : array_like
        The shock normal vector.
    """

    global normal_eq_input
    normal_eq_input['A'] = vec_1[0]
    normal_eq_input['B'] = vec_1[1]
    normal_eq_input['C'] = vec_1[2]
    normal_eq_input['D'] = vec_2[0]
    normal_eq_input['E'] = vec_2[1]
    normal_eq_input['F'] = vec_2[2]

    n_start = np.array([1.0, 0.0, 0.0])
    normal = fsolve(normal_equation, n_start, xtol=1e-13)

    return normal


def normal_equation(N):
    """The set of equations used in finding the shock normal vector.

    This function defines three equations that are used to determine the
    shock normal vector. The first two correspond to dot products
    between the input vector and two other vectors (these are given as
    inputs to solve_normal_eq), and the third one is required to make
    sure that the solution is a unit vector.

    Parameters
    ----------
    N : array_like
        Input vector

    Returns
    -------
    eqns : list
        Three constraining equations in a list.
    """

    global normal_eq_input
    A = normal_eq_input['A']
    B = normal_eq_input['B']
    C = normal_eq_input['C']
    D = normal_eq_input['D']
    E = normal_eq_input['E']
    F = normal_eq_input['F']

    return [
        A * N[0] + B * N[1] + C * N[2],
        D * N[0] + E * N[1] + F * N[2],
        N[0] ** 2 + N[1] ** 2 + N[2] ** 2 - 1.0
    ]


def normal_sign(normal, shock_type, V_vector_up):
    r"""Determine the sign, i.e., the direction, of the shock normal.

    This functions determines the sign of the shock normal vector such
    that :math:`V_{\mathrm{up}}\cdot\hat{n} \geq 0` for a fast-forward
    (FF) shock and :math:`V_{\mathrm{up}}\cdot\hat{n} \leq 0` for a
    fast-reverse (FR) shock.

    Parameters
    ----------
    normal : array_like
        Shock normal vector.
    shock_type : int
        Shock type, 1 for an FF-shock and 2 for an FR-shock.
    V_vector_up : array_like
        Upstream plasma velocity vector.

    Returns
    -------
    sign * normal : array_like
        Correctly signed shock normal vector.
    """

    sign = 1

    if ((shock_type == 1 and np.dot(normal, V_vector_up) < 0)
            or (shock_type == 2 and np.dot(normal, V_vector_up) > 0)):
        sign = -1

    return sign * normal


def shock_normal(
        shock_type, B_vector_up, B_vector_down,
        V_vector_up, V_vector_down, method):
    """Calculate the shock normal vector.

    This function first calculates the shock normal using one of three
    available methods and then determines the correct sign for the
    normal. The function raises an error if the shock normal cannot be
    determined successfully.

    Parameters
    ----------
    shock_type : int
        Shock type, 1 for an FF-shock and 2 for an FR-shock.
    B_vector_up : array_like
        Upstream magnetic field vector.
    B_vector_down : array_like
        Downstream magnetic field vector.
    V_vector_up : array_like
        Upstream plasma velocity vector.
    V_vector_down : array_like
        Downstream plasma velocity vector.
    method : int
        Shock normal calculation method ID: 0 = MX3, 1 = MFC,
        2 = MX1 + MX2 average, 3 = MVA (not implemented). See IPShocks
        documentation for details.

    Returns
    -------
    normal : array_like
        The shock normal vector.
    """

    B_vector_up = np.array(B_vector_up)
    B_vector_down = np.array(B_vector_down)
    V_vector_up = np.array(V_vector_up)
    V_vector_down = np.array(V_vector_down)

    check = 0

    if method == 0:
        vec_1 = np.cross(
            B_vector_down - B_vector_up, V_vector_down - V_vector_up)
        vec_2 = B_vector_down - B_vector_up
        normal = solve_normal_eq(vec_1, vec_2)
        normal = normal_sign(normal, shock_type, V_vector_up)
    elif method == 1:
        vec_1 = np.cross(B_vector_down, B_vector_up)
        vec_2 = B_vector_down - B_vector_up
        normal = solve_normal_eq(vec_1, vec_2)
        normal = normal_sign(normal, shock_type, V_vector_up)
    elif method == 2:
        normals = np.zeros((3, 2))
        checks = np.zeros(2)
        for j in range(2):
            vec_1 = np.cross(B_vector_down, (V_vector_down - V_vector_up))
            if j == 1:
                vec_1 = np.cross(B_vector_up, (V_vector_down - V_vector_up))
            vec_2 = B_vector_down - B_vector_up
            normal = solve_normal_eq(vec_1, vec_2)
            normals[:, j] = normal_sign(normal, shock_type, V_vector_up)
        normal = np.mean(normals, axis=1)
        normal /= np.linalg.norm(normal)
        check = np.sum(checks)
    elif method == 3:
        return np.array([1.0, 0.0, 0.0])
    
    if check != 0:
        print(
            "The method to calculate the normal vector did not converge. "
            "The program stops.")
        raise RuntimeError("Convergence issue in solving the normal vector.")
    
    return normal


# Initialize a global variable to store the reference to the vertical
# line in the plotting windows
last_vline = []
unclear = False


def on_key(event):
    """Function to handle key press events when evaluating a shock plot.

    During the analysis run a shock plot is displayed. During this the
    user can fine tune the shock time as well as mark the shock down as
    an unclear shock. This function enables the user to do these tasks
    using the arrow keys.

    Parameters
    ----------
    event : KeyEvent
        Key press event, left or right to shift the shock time one
        second back or forward, respectively, down to save the current
        shock time, and up to mark the event as unclear.
    """

    global last_vline, last_click_time, time_adjusted, unclear

    line_kwargs = dict(color='blue', linestyle='--', linewidth=1)

    for ax in axs:
        for annotation in ax.texts:
            annotation.remove()

    if event.key == 'right':
        if last_click_time:
            new_time = last_click_time + timedelta(seconds=1)

            # Remove previous vertical lines
            if last_vline:
                for line in last_vline:
                    line.remove()
                last_vline = []

            # Draw a new vertical line at the updated time
            last_vline = []
            for ax in axs:
                vline = ax.axvline(new_time, **line_kwargs)
                last_vline.append(vline)

            # Update the last click time
            last_click_time = new_time

            fig.canvas.draw()
    elif event.key == 'left':
        if last_click_time:
            new_time = last_click_time - timedelta(seconds=1)

            # Remove previous vertical lines
            if last_vline:
                for line in last_vline:
                    line.remove()
                last_vline = []
            
            # Draw a new vertical line at the updated time
            last_vline = []
            for ax in axs:
                vline = ax.axvline(new_time, **line_kwargs)
                last_vline.append(vline)

            # Update the last click time
            last_click_time = new_time

            fig.canvas.draw()
    elif event.key == 'down':
        time_adjusted = True
        if last_click_time:
            formatted_final_time = last_click_time.strftime(
                '%Y    %m    %d    %H    %M    %S')

            with open(shock_times_out_fname, 'a') as f:
                f.write(formatted_final_time + '\n')

            # Remove the previous vertical line if it exists
            if last_vline:
                for line in last_vline:
                    line.remove()
                last_vline = []

            # Draw a new vertical line starting from the shock time
            last_vline = []
            for ax in axs:
                vline = ax.axvline(
                    last_click_time, color='black', linestyle='--',
                    linewidth=1)
                last_vline.append(vline)

            text_to_print = f"Time saved:\n{formatted_final_time}"
            
            # Add a new annotation above the title in the top subplot
            axs[0].annotate(
                text_to_print, xy=(0.5, 1.3), xycoords='axes fraction',
                ha='center', va='center', fontsize=12)

            fig.canvas.draw()
    elif event.key == 'up':
        unclear = True
        text_to_print = "Event marked as unclear"
            
        # Add a new annotation above the title in the top subplot
        axs[0].annotate(
            text_to_print, xy=(0.5, 1.3), xycoords='axes fraction',
            ha='center', va='center', fontsize=12)

        fig.canvas.draw()


# ----------------------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Advanced settings (see documentation before changing these)
# ----------------------------------------------------------------------

# Shock time data file
shock_times_fname = 'shocks.dat'

# Shock time output file. If the shock times are preliminary, better
# estimates are saved here
shock_times_out_fname = 'shocks_out.dat'

# Output file of the analysis results
analysis_output_fname = 'shock_parameters.dat'

# The default and an alternative method for determining the shock
# normal. The method IDs are: 0 = MX3, 1 = MFC, 2 = MX1 + MX2 average,
# 3 = MVA (not implemented)
orig_normal_method_id = 0
additional_normal_method_id = 1

# Directory for saving the shock plots and the CSV file where the final
# results for clear shocks are saved to
plot_directory = 'clear_shock_plots'
csv_file = 'clear_shock_parameters.csv'

# Directory for saving the unclear shock plots and the CSV file where
# the final results for unclear shocks are saved to
unclear_plot_directory = 'unclear_shock_plots'
unclear_csv_file = 'unclear_shock_parameters.csv'

# ----------------------------------------------------------------------
# Reading the input file
# ----------------------------------------------------------------------

with open(shock_times_fname, 'r') as file1:
    lines = file1.readlines()
    N_sh = len(lines) - 11  # The number of shock events

    # Initialize the header options
    comment11 = ''
    comment12 = ''
    comment13 = ''
    sc = 0
    comment2 = ''
    plot_events = 0
    comment31 = ''
    comment32 = ''
    comment33 = ''
    filter_line = 0
    comment4 = ''

    # Read the values for the header options from the opened input file
    for i, line in enumerate(lines):
        if i == 0:
            comment11 = line.strip()
        elif i == 1:
            comment12 = line.strip()
        elif i == 2:
            comment13 = line.strip()
        elif i == 3:
            sc = int(line.strip())
        elif i == 4:
            comment2 = line.strip()
        elif i == 5:
            plot_events = int(line.strip())
        elif i == 6:
            comment31 = line.strip()
        elif i == 7:
            comment32 = line.strip()
        elif i == 8:
            comment33 = line.strip()
        elif i == 9:
            if len(line.strip()) == 1:
                filter_line = int(line.strip())
            else:
                filter_line = line.strip()
        elif i == 10:
            comment4 = line.strip()

    # Check which filter option was chosen
    if filter_line == 0:  # No filtering
        filter_options = 0
    elif filter_line == 1:  # Default filter
        filter_options = 1
    else:  # User-defined four value filter
        filter_options = list(map(float, filter_line.split(',')))

    # Initialize lists where the candidate shock times will be stored
    data = [[], [], [], [], [], []]  # Raw string data from the file
    shock_datetimes = []  # Datetime objects
    shock_formatted_times = []  # Times as formatted strings
    shock_timestamps = []  # Seconds since January 1, 1970 (Unix epoch)

    # Read the shock times from input text file
    for i in range(11, len(lines)):
        line = lines[i]
        year, month, day, hour, minute, second = line.split()

        data[0].append(year)
        data[1].append(month)
        data[2].append(day)
        data[3].append(hour)
        data[4].append(minute)
        data[5].append(second)

        datetime_format = datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second))
        shock_datetimes.append(datetime_format)

        shock_formatted_times.append(
            datetime_format.strftime('%Y-%m-%d/%H:%M'))

    # Change the time format to seconds since January 1, 1970
    shock_timestamps = [
        (dt - datetime(1970, 1, 1)).total_seconds() for dt in shock_datetimes]

# ----------------------------------------------------------------------
# Initializing the output of the analysis
# ----------------------------------------------------------------------

# Initialize the shock time output file
with open(shock_times_out_fname, 'w') as file2:
    file2.write(comment11 + '\n')
    file2.write(f'{"": >15}' + comment12 + '\n')
    file2.write(f'{"": >15}' + comment13 + '\n')
    file2.write(str(sc) + '\n')
    file2.write(comment2 + '\n')
    file2.write(str(plot_events) + '\n')
    file2.write(comment31 + '\n')
    file2.write(comment32 + '\n')
    file2.write(comment33 + '\n')
    file2.write(str(filter_line) + '\n')
    file2.write(comment4 + '\n')

# Initialize the shock parameter file and write the header to it
header = (
    "Year Month   Day  Hour   Min   Sec   Type       "
    "Position                         B_up                    "
    "Bx_up                   By_up                   "
    "Bz_up                   B_down                  "
    "Bx_down                 By_down                 "
    "Bz_down                 B_ratio               "
    "V_up                    Vx_up                   "
    "Vy_up                 Vz_up                "
    "V_down                  Vx_down                 "
    "Vy_down               Vz_down               "
    "V_jump                 Np_up                "
    "Np_down                Np_ratio              "
    "Tp_up                 Tp_down               "
    "Tp_ratio             Cs_up                 "
    "Va_up                 Vms_up                 "
    "Beta_up               "
    "n_vector                                                         "
    "Theta                 Vsh                     "
    "M_A                   Mms                   "
    "V_quality "
    "Analysis int length "
    "Mag data res "
    "Pla data res")

with open(analysis_output_fname, 'w') as file3:
    file3.write(header + '\n')

# Initialize the CSV file's header line
header_line_csv = (
    "year,month,day,hour,minute,second,type,"
    "spacecraft_x,spacecraft_y,spacecraft_z,"
    "magnetic_field_upstream,error_magnetic_field_upstream,"
    "magnetic_field_upstream_x,error_magnetic_field_upstream_x,"
    "magnetic_field_upstream_y,error_magnetic_field_upstream_y,"
    "magnetic_field_upstream_z,error_magnetic_field_upstream_z,"
    "magnetic_field_downstream,error_magnetic_field_downstream,"
    "magnetic_field_downstream_x,error_magnetic_field_downstream_x,"
    "magnetic_field_downstream_y,error_magnetic_field_downstream_y,"
    "magnetic_field_downstream_z,error_magnetic_field_downstream_z,"
    "magnetic_field_ratio,error_magnetic_field_ratio,"
    "speed_upstream,error_speed_upstream,"
    "velocity_upstream_x,error_velocity_upstream_x,"
    "velocity_upstream_y,error_velocity_upstream_y,"
    "velocity_upstream_z,error_velocity_upstream_z,"
    "speed_downstream,error_speed_downstream,"
    "velocity_downstream_x,error_velocity_downstream_x,"
    "velocity_downstream_y,error_velocity_downstream_y,"
    "velocity_downstream_z,error_velocity_downstream_z,"
    "speed_jump,error_speed_jump,"
    "proton_density_upstream,error_proton_density_upstream,"
    "proton_density_downstream,error_proton_density_downstream,"
    "proton_density_ratio,error_proton_density_ratio,"
    "proton_temperature_upstream,error_proton_temperature_upstream,"
    "proton_temperature_downstream,error_proton_temperature_downstream,"
    "proton_temperature_ratio,error_proton_temperature_ratio,"
    "sound_speed_upstream,error_sound_speed_upstream,"
    "alfven_speed_upstream,error_alfven_speed_upstream,"
    "magnetosonic_speed_upstream,error_magnetosonic_speed_upstream,"
    "plasma_beta_upstream,error_plasma_beta_upstream,"
    "normal_x,error_normal_x,"
    "normal_y,error_normal_y,"
    "normal_z,error_normal_z,"
    "theta_upstream,error_theta_upstream,"
    "shock_speed,error_shock_speed,"
    "alfven_mach_number,error_alfven_mach_number,"
    "magnetosonic_mach_number,error_magnetosonic_mach_number,"
    "is_radial_velocity,"
    "analysis_interval,"
    "magnetic_field_resolution,"
    "plasma_resolution")

# Write the header to the CSV files for both the clear and unclear
# shocks
with open(csv_file, 'w') as file3:
    file3.write(header_line_csv + '\n')

with open(unclear_csv_file, 'w') as file4:
    file4.write(header_line_csv + '\n')

# Initialize a list for events with insufficient data
bad_events = []

# ----------------------------------------------------------------------
# Looping through all the shock candidate times
# ----------------------------------------------------------------------

for i in range(0, N_sh):
    # For run time duration estimation
    loop_start_time = time.time()

    # Initializing Booleans for the shock event
    time_adjusted = False
    unclear = False

    # ------------------------------------------------------------------
    # Downloading and processing data for the analysis
    # ------------------------------------------------------------------

    (mag_dataframe, pla_dataframe, sc_pos, output_add,
     shock_timestamp_new, pla_bin_rads) = download_and_process_data(
        shock_datetimes[i], sc, filter_options)

    # ------------------------------------------------------------------
    # Updating the shock time
    # ------------------------------------------------------------------

    shock_timestamps[i] = shock_timestamp_new

    # ------------------------------------------------------------------
    # Determining the shock type (FF or FR) as well as the upstream and
    # downstream mean values.
    # ------------------------------------------------------------------

    mag_vars = ['B', 'Bx', 'By', 'Bz']
    pla_vars = ['Vx', 'Vy', 'Vz', 'V', 'Np', 'Tp']

    # Determine the analysis intervals upstream and downstream of the
    # shock. Note that these depend on the shock type (FF or FR).

    # Length of the analysis intervals in seconds (default 8 minutes)
    length = 8 * 60

    # Maximum possible distance from the shock time (30 minutes)
    length_limit = 30 * 60

    # Ulysses, Voyager 1, and Voyager 2 have a different length limits
    # for the analysis intervals (12 minutes for Ulysses and 36 minutes
    # Voyager 1 and 2)
    if sc == 6:
        length = 12 * 60
    elif sc in [11, 12]:
        length_limit = 36 * 60

    # Intervals excluded upstream (1 minute) and downstream (2 minutes)
    # of the shock
    upstream_gap = 1 * 60
    downstream_gap = 2 * 60

    # Initialize the Helios data flag, 0 for 6 sec magnetic field data
    # and 1 for 40.5 sec magnetic field data
    helios_dataset_flag = 0

    # Next: Checking that there is enough datapoints around the shock.
    # If not, the length of the analysis interval is increased until
    # either the upper limit for the interval length is reached or
    # upstream and downstream intervals both contain at least 3 data
    # points

    # Test iteratively if the shock fulfils the criteria of an FF shock
    # (loop index 0) or an FR shock (loop index 1)
    for j in range(1, -1, -1):
        too_few_mag_points = 0
        too_few_pla_points = 0

        # Furthest points of the upstream and downstream intervals from
        # the shock
        upstream_furthest = upstream_gap + length
        downstream_furthest = downstream_gap + length

        # Other spacecraft
        if sc not in [4, 5, 6, 11, 12]:
            t_up_both = np.array([
                [shock_timestamps[i] - upstream_furthest,
                 shock_timestamps[i] - upstream_gap],
                [shock_timestamps[i] + upstream_gap,
                 shock_timestamps[i] + upstream_furthest]])
            t_down_both = np.array([
                [shock_timestamps[i] + downstream_gap,
                 shock_timestamps[i] + downstream_furthest],
                [shock_timestamps[i] - downstream_furthest,
                 shock_timestamps[i] - downstream_gap]])
        # Helios, Ulysses and Voyager
        elif sc in [4, 5, 6, 11, 12]:
            while abs(upstream_furthest - upstream_gap) <= length_limit:
                t_up_both = np.array([
                    [shock_timestamps[i] - upstream_furthest,
                     shock_timestamps[i] - upstream_gap],
                    [shock_timestamps[i] + upstream_gap,
                     shock_timestamps[i] + upstream_furthest]])
                t_down_both = np.array([
                    [shock_timestamps[i] + downstream_gap,
                     shock_timestamps[i] + downstream_furthest],
                    [shock_timestamps[i] - downstream_furthest,
                     shock_timestamps[i] - downstream_gap]])

                t_up = t_up_both[j]
                t_down = t_down_both[j]

                too_few_mag_points = analysis_interval_check(
                    mag_dataframe, mag_vars, t_up, t_down)
                too_few_pla_points = analysis_interval_check(
                    pla_dataframe, pla_vars, t_up, t_down)

                # Increase the interval length if the current interval
                # contains less than 3 data points.
                if too_few_mag_points or too_few_pla_points:
                    # Change the magnetic field data source for Helios
                    # if there were not enough data points. Replace the
                    # magnetic field data with the additional data
                    # (i.e., the magnetic field data from the plasma
                    # product datasets)
                    if (sc in [4, 5]
                            and (helios_dataset_flag == 0)
                            and (too_few_mag_points == 1)):
                        helios_dataset_flag = 1

                        add_dataframe = output_add[0]
                        shock_timestamps[i] = output_add[1]
                        sc_pos = output_add[2]

                        mag_dataframe['EPOCH'] = add_dataframe['EPOCH']
                        mag_dataframe['B'] = add_dataframe['B']
                        mag_dataframe['Bx'] = add_dataframe['Bx']
                        mag_dataframe['By'] = add_dataframe['By']
                        mag_dataframe['Bz'] = add_dataframe['Bz']

                    # Increase the interval by the minimum resolution
                    # of the data product which did not have enough
                    # data points.
                    if too_few_pla_points == 0:
                        increment = return_resolution(
                            mag_vars[0], sc, helios_dataset_flag)
                    else:
                        increment = return_resolution(
                            pla_vars[0], sc, helios_dataset_flag)

                    upstream_furthest += increment
                    downstream_furthest += increment
                else:
                    break

        # Jump straight to FR-type check if there is not enough data
        if ((too_few_mag_points == 1 or too_few_pla_points == 1)
                and j == 1):
            continue

        # If the FR-type check does not contain enough data points,
        # reject the whole event
        jump_to_plotting = False
        if ((too_few_mag_points == 1 or too_few_pla_points == 1)
                and j == 0):
            bad_events.append(shock_timestamps[i])
            analysis_int_length = 30
            shock_type = 'N'
            jump_to_plotting = True
            break

        # The analysis intervals used for calculating the upstream and
        # downstream mean values
        t_up = t_up_both[j]
        t_down = t_down_both[j]

        # Length of the analysis interval in minutes
        analysis_int_length = (t_up[1] - t_up[0]) / 60

        # Calculating the mean and standard deviation values of the
        # parameters in the upstream and downstream intervals
        mag_up_dict = mean_data(mag_dataframe, mag_vars, t_up)
        mag_down_dict = mean_data(mag_dataframe, mag_vars, t_down)
        pla_up_dict = mean_data(pla_dataframe, pla_vars, t_up)
        pla_down_dict = mean_data(pla_dataframe, pla_vars, t_down)

        # The mean values of the parameters assigned explicitly
        B_mean_up = mag_up_dict[0]['B']
        B_mean_down = mag_down_dict[0]['B']
        B_vector_up = [
            mag_up_dict[0]['Bx'], mag_up_dict[0]['By'], mag_up_dict[0]['Bz']]
        B_vector_down = [
            mag_down_dict[0]['Bx'], mag_down_dict[0]['By'],
            mag_down_dict[0]['Bz']]
        Np_mean_up = pla_up_dict[0]['Np']
        Np_mean_down = pla_down_dict[0]['Np']
        Tp_mean_up = pla_up_dict[0]['Tp']
        Tp_mean_down = pla_down_dict[0]['Tp']
        V_mean_up = pla_up_dict[0]['V']
        V_mean_down = pla_down_dict[0]['V']
        V_vector_up = [
            pla_up_dict[0]['Vx'], pla_up_dict[0]['Vy'], pla_up_dict[0]['Vz']]
        V_vector_down = [
            pla_down_dict[0]['Vx'], pla_down_dict[0]['Vy'],
            pla_down_dict[0]['Vz']]

        B_ratio = B_mean_down / B_mean_up
        Np_ratio = Np_mean_down / Np_mean_up
        Tp_ratio = Tp_mean_down / Tp_mean_up
        V_jump = abs(V_mean_down - V_mean_up)

        # Shock criteria (type: 0 = not a shock, 1 = FF-shock,
        # 2 = FR-shock)
        type = 0

        if (B_ratio >= 1.2) and (Np_ratio >= 1.2) and (Tp_ratio > 1 / 1.2):
            if (j == 0) and ((V_mean_down - V_mean_up) >= 20):  # FF shock
                type = 1
                break
            if (j == 1) and ((V_mean_up - V_mean_down) >= 20):  # FR shock
                type = 2
                break

    if not jump_to_plotting:
        # If the variable plot_events is set to 0 in the input file,
        # then analysis is not performed any further for events which
        # are not FF- or FR-shocks. There will be no output or plots
        # produced of these events
        if (type == 0) and (plot_events == 0):
            continue

        # --------------------------------------------------------------
        # Checking the quality of the velocity vectors in the upstream
        # and downstream intervals
        # --------------------------------------------------------------

        # Note regarding velocity data for STEREO-A and STEREO-B:
        # If there is no velocity data to determine the velocity vector
        # upstream and/or downstream then the velocity vector is
        # replaced by a radial vector with the magnitude of the bulk
        # speed, i.e., V

        bad_vel = 0
        # Filter the DataFrame for t_up range and check for finite values
        df_up = pla_dataframe[
            (pla_dataframe['EPOCH'] >= t_up[0]) & (pla_dataframe['EPOCH'] <= t_up[1])]
        df_down = pla_dataframe[
            (pla_dataframe['EPOCH'] >= t_down[0]) & (pla_dataframe['EPOCH'] <= t_down[1])]
        finite_up = df_up[['Vx', 'Vy', 'Vz']].notna().all(axis=1)
        finite_down = df_down[['Vx', 'Vy', 'Vz']].notna().all(axis=1)
        gg_up = df_up.index[finite_up].tolist()
        gg_down = df_down.index[finite_down].tolist()

        # Update bad_vel if any of the resulting indices are empty
        if len(gg_up) == 0 or len(gg_down) == 0:
            bad_vel = 1

        if bad_vel == 1 and sc in [2, 3]:
            # Change the time series variables
            pla_pnts = len(pla_dataframe['EPOCH'])
            pla_dataframe['Vx'] = pla_dataframe['V']
            pla_dataframe['Vy'] = np.zeros(pla_pnts)
            pla_dataframe['Vz'] = np.zeros(pla_pnts)

            # Fix tplot time series to match this
            V_vector_up = np.array([V_mean_up, 0, 0])
            V_vector_down = np.array([V_mean_down, 0, 0])

            pla_up_dict[0]['Vx'] = V_mean_up
            pla_up_dict[0]['Vy'] = V_mean_up
            pla_up_dict[0]['Vz'] = V_mean_up

            pla_down_dict[0]['Vx'] = V_mean_down
            pla_down_dict[0]['Vy'] = V_mean_down
            pla_down_dict[0]['Vz'] = V_mean_down

            pla_up_dict[1]['Vx'] = pla_up_dict[1]['V']
            pla_down_dict[1]['Vx'] = pla_down_dict[1]['V']

            pla_up_dict[0]['Vy'] = 0
            pla_up_dict[0]['Vz'] = 0

            pla_up_dict[1]['Vy'] = 0
            pla_up_dict[1]['Vz'] = 0

            pla_down_dict[0]['Vy'] = 0
            pla_down_dict[0]['Vz'] = 0

            pla_down_dict[1]['Vy'] = 0
            pla_down_dict[1]['Vz'] = 0

        # --------------------------------------------------------------
        # Calculating the mean resolution of the magnetic field and
        # plasma data over the analysis intervals
        # --------------------------------------------------------------

        avg_res_mag = calculate_mean_resolution(
            mag_dataframe, mag_vars, t_up, t_down, sc, helios_dataset_flag)
        avg_res_pla = calculate_mean_resolution(
            pla_dataframe, pla_vars, t_up, t_down, sc, helios_dataset_flag)

        if not avg_res_mag or not avg_res_pla:
            print("There is a problem calculating the mean data resolutions.")
            continue

        # --------------------------------------------------------------
        # Calculating the mean values of plasma characteristics (sound
        # speed, Alfvén speed, magnetosonic speed, and plasma beta)
        # upstream of the shock
        # --------------------------------------------------------------

        # Averaging bin radii of plasma data for resampling
        if sc == 0:
            bin_rad = 32.0  # i.e. 64 / 2 --> (+- 32sec)
        if sc == 1:
            bin_rad = pla_bin_rads  # Determined in the downloaded data
        if (sc == 2) or (sc == 3):
            bin_rad = 30.0
        if (sc == 4) or (sc == 5):
            bin_rad = 20.25
        if sc == 6:
            bin_rad = 60.0
        if sc == 10:
            bin_rad = 30.0
        if (sc == 11) or (sc == 12):
            bin_rad = 6.0
        if sc == 13:
            bin_rad = 30.0
        if sc == 14:
            bin_rad = 30.0
        if sc == 15:
            bin_rad = 2

        # Extract values of Np and Tp in the interval
        Np = pla_dataframe[(pla_dataframe['EPOCH'] >= t_up[0]) & (
                    pla_dataframe['EPOCH'] <= t_up[1])]['Np']
        Tp = pla_dataframe[(pla_dataframe['EPOCH'] >= t_up[0]) & (
                    pla_dataframe['EPOCH'] <= t_up[1])]['Tp']
        V = pla_dataframe[(pla_dataframe['EPOCH'] >= t_up[0]) & (
                    pla_dataframe['EPOCH'] <= t_up[1])]['V']
        Bt = mag_dataframe[(mag_dataframe['EPOCH'] >= t_up[0]) & (
                    mag_dataframe['EPOCH'] <= t_up[1])]['B']

        # Magnetic field data is resampled to the plasma data resolution
        # it has a higher resolution
        if (sc in [0, 1, 2, 3, 6, 11, 12, 13, 15]
                or (sc in [4, 5] and helios_dataset_flag == 0)):
            B_rs = resample(
                mag_dataframe['B'], mag_dataframe['EPOCH'],
                pla_dataframe['EPOCH'], t_up, sc, bin_rad)
            B_is_averaged = 1
        # For PSP the plasma data is higher resolution than magnetic
        # field data
        elif sc == 14:
            Np = resample(
                pla_dataframe['Np'], pla_dataframe['EPOCH'],
                mag_dataframe['EPOCH'], t_up, sc, bin_rad)
            Tp = resample(
                pla_dataframe['Tp'], pla_dataframe['EPOCH'],
                mag_dataframe['EPOCH'], t_up, sc, bin_rad)
            V = resample(
                pla_dataframe['V'], pla_dataframe['EPOCH'],
                mag_dataframe['EPOCH'], t_up, sc, bin_rad)
            B_rs = Bt
        else:
            # For Cluster only linear interpolation is required since
            # time tags are almost the same (difference in order of ms)
            if sc in [7, 8, 9]:
                B_rs = np.interp(
                    pla_dataframe[(pla_dataframe['EPOCH'] >= t_up[0]) & (pla_dataframe['EPOCH'] <= t_up[1])]['EPOCH'],mag_dataframe['EPOCH'], mag_dataframe['B'])
            else:
                B_rs = mag_dataframe[
                    (mag_dataframe['EPOCH'] >= t_up[0]) & (mag_dataframe['EPOCH'] <= t_up[1])]['B']
                B_is_averaged = 0

        # Constants
        u0 = 12.56637e-7  # T.m/A  permeability of free space,
        mp = 1.67262158e-27  # Kg proton mass
        kB = 1.3806488e-23  # m^2.Kg/s^2.K ; Boltzmann constant
        gamma = 5.0 / 3.0  # Gamma factor
        ep_ratio = 2.0  # Electron-to-proton temperature ratio (NOT USED!!!)

        # Radial distance for electron temp calculations
        r_AU = 1

        # Spacecraft other than Helios, Ulysses, Voyager, PSP, and SolO
        # are approximated to be at 1 AU
        if sc in [4, 5, 6, 11, 12, 14, 15]:
            r_AU = np.abs(sc_pos.iloc[0])

        A = 1.462768889297766 * 1e+05
        B = -0.664193092275818

        # Creating an electron temperature array (depends on the radial
        # distance from the Sun)
        Te = np.full(len(Tp), A * r_AU ** B)

        # Standard deviation (estimated error) of the electron
        # temperature. Determined as the estimated standard deviation of
        # the Te(R_AU)-fit
        Te_std = 4.971842720311243 * 1e+04

        # Plasma characteristics
        B2 = B_rs * 1e-9
        Np2 = Np * 1e+6
        Cs = (((gamma*kB)*(Tp + Te))/mp)**0.5/ 1000
        V_A = (B2 / (np.sqrt(u0 * mp * Np2))) / 1000  # Alfven speed
        Vms = np.sqrt(V_A**2 + Cs**2)  # Magnetosonic speed
        Dp = Np * V**2 * 1.67e-6 * 1.16  # Dynamic pressure
        beta = (2*Np2*kB*(Tp + Te)*u0)/B2**2  # Plasma beta

        # Form dataframe time series for these parameters
        # Extract timestamps from 'EPOCH' column of pla_dataframe
        t_ax = pla_dataframe['EPOCH']
        if sc in [14]:  # Lower res t_ax is the mag t_ax with PSP
            t_ax = mag_dataframe['EPOCH']

        # Filter timestamps
        t_ax = t_ax[(t_ax >= t_up[0]) & (t_ax <= t_up[1])]

        # Create DataFrame
        pla_char_dataframe = pd.DataFrame({
            'EPOCH': t_ax,
            'Cs': Cs,
            'V_A': V_A,
            'Vms': Vms,
            'beta': beta,
            'Dp': Dp})

        # Calculate the mean values and standard deviations in the upstream area
        pla_char_up_dict = mean_data(
            pla_char_dataframe, ['Cs', 'V_A', 'Vms', 'beta', 'Dp'], t_up)

        # For error analysis purposes calculate also other means
        # (basically partial derivatives)
        doo_Cs_doo_Te = kB / (2 * mp) * np.nanmean(
            (1000. * pla_char_dataframe['Cs'])**(-1)) / 1000
        doo_Vms_doo_Te = kB / (2 * mp) * np.nanmean(
            (1000. * pla_char_dataframe['Vms'])**(-1)) / 1000
        doo_beta_doo_Te = np.nanmean(2 * u0 * kB * Np2 / B2**2)

        # Extract the mean values
        Cs_mean_up = pla_char_up_dict[0]['Cs']
        V_A_mean_up = pla_char_up_dict[0]['V_A']
        Vms_mean_up = pla_char_up_dict[0]['Vms']
        beta_mean_up = pla_char_up_dict[0]['beta']
        Dp_mean_up = pla_char_up_dict[0]['Dp']

        # --------------------------------------------------------------
        # Calculating the normal vector of the shock
        # --------------------------------------------------------------

        # Method IDs:
        #   0 = MX3,
        #   1 = MFC,
        #   2 = MX1 + MX2 average,
        #   3 = MVA (not implemented)

        normal_method_id = orig_normal_method_id

        # If there is no velocity vector data additional method is used
        if bad_vel == 1:
            normal_method_id = additional_normal_method_id

        normal_eq_input = {}

        normal = shock_normal(
            type, B_vector_up, B_vector_down,
            V_vector_up, V_vector_down,
            normal_method_id)

        # --------------------------------------------------------------
        # WILL BE REMOVED IN THE FUTURE
        # normal_method_id = 3: use Minimum Variance Analysis of
        # magnetic field data to determine the normal (NOT IMPLEMENTED
        # AT THE MOMENT)
        # if normal_method_id == 3:
        #     This is from the IDL code
        #     RESOLVE_ROUTINE, 'mva_normal'
        #     mva_normal, shock_timestamps, SC, normal, eigs, an_int, e_ratio
        #     normal = normal_sign(normal, type, V_vector_up)

        # Double check that the normal vector given by the MX3 method is
        # perpendicular to all the correct vectors
        B_vector_down = np.array(B_vector_down)
        B_vector_up = np.array(B_vector_up)
        V_vector_down = np.array(V_vector_down)
        V_vector_up = np.array(V_vector_up)

        delta_B = B_vector_down - B_vector_up
        delta_V = V_vector_down - V_vector_up

        if ((np.dot(delta_B, normal) > 1e-12
             or np.dot(np.cross(delta_B, delta_V), normal) > 1e-12)
                and normal_method_id == 0):
            raise ValueError(
                "The normal vector calculated using the MX3 method is not "
                "correct: the vector should be perpendicular to (ΔV x ΔB) "
                "and ΔB")

        # --------------------------------------------------------------------------
        # Calculating the shock speed
        # --------------------------------------------------------------------------

        # Calculate the flux vector
        flux = (Np_mean_down*V_vector_down - Np_mean_up*V_vector_up)/(Np_mean_down - Np_mean_up)

        # Calculate V_shock as the absolute value of the dot product of the flux vector with the normal vector
        V_shock = abs(np.dot(flux, normal))

        # --------------------------------------------------------------------------
        # The Alfvén and magnetosonic Mach numbers
        # --------------------------------------------------------------------------

        # Transformation to the rest frame of the shock depends on the shock type
        Vsh_trans = V_shock
        if type == 2:
            Vsh_trans = -V_shock

        # Calculate Mach numbers
        Mms_up = abs(np.dot(V_vector_up, normal) - Vsh_trans) / Vms_mean_up
        M_A_up = abs(np.dot(V_vector_up, normal) - Vsh_trans) / V_A_mean_up

        # --------------------------------------------------------------------------
        # Calculation of shock theta
        # --------------------------------------------------------------------------

        # Calculate the cosine of the angle between B_vector_up and the normal
        CosTheta = abs(np.dot(B_vector_up, normal)) / (
                    np.linalg.norm(B_vector_up) * np.linalg.norm(normal))

        # Calculate the angle in degrees
        shock_theta = np.arccos(CosTheta) * 180 / np.pi

        # --------------------------------------------------------------------------
        # Error analysis
        # --------------------------------------------------------------------------

        shock_type = 'forward'
        if type == 2:
            shock_type = 'reverse'

        mag_up = [mag_up_dict[0]["B"],
                  mag_up_dict[0]["Bx"],
                  mag_up_dict[0]["By"],
                  mag_up_dict[0]["Bz"]
                  ]
        mag_up_std = [mag_up_dict[1]["B"],
                      mag_up_dict[1]["Bx"],
                      mag_up_dict[1]["By"],
                      mag_up_dict[1]["Bz"]]
        mag_down = [mag_down_dict[0]["B"],
                    mag_down_dict[0]["Bx"],
                    mag_down_dict[0]["By"],
                    mag_down_dict[0]["Bz"]
                    ]
        mag_down_std = [mag_down_dict[1]["B"],
                        mag_down_dict[1]["Bx"],
                        mag_down_dict[1]["By"],
                        mag_down_dict[1]["Bz"]]

        pla_up = [
            pla_up_dict[0]["Np"],
            pla_up_dict[0]["Tp"],
            pla_up_dict[0]["V"],
            pla_up_dict[0]["Vx"],
            pla_up_dict[0]["Vy"],
            pla_up_dict[0]["Vz"]
        ]

        pla_up_std = [pla_up_dict[1]["Np"],
                      pla_up_dict[1]["Tp"],
                      pla_up_dict[1]["V"],
                      pla_up_dict[1]["Vx"],
                      pla_up_dict[1]["Vy"],
                      pla_up_dict[1]["Vz"]
                      ]

        pla_down = [
            pla_down_dict[0]["Np"],
            pla_down_dict[0]["Tp"],
            pla_down_dict[0]["V"],
            pla_down_dict[0]["Vx"],
            pla_down_dict[0]["Vy"],
            pla_down_dict[0]["Vz"]
        ]
        pla_down_std = [pla_down_dict[1]["Np"],
                        pla_down_dict[1]["Tp"],
                        pla_down_dict[1]["V"],
                        pla_down_dict[1]["Vx"],
                        pla_down_dict[1]["Vy"],
                        pla_down_dict[1]["Vz"]]

        char_up = [pla_char_up_dict[0]["Cs"],
                   pla_char_up_dict[0]["V_A"],
                   pla_char_up_dict[0]["Vms"],
                   pla_char_up_dict[0]["beta"],
                   pla_char_up_dict[0]["Dp"]
                   ]

        char_std_up = [pla_char_up_dict[1]["Cs"],
                       pla_char_up_dict[1]["V_A"],
                       pla_char_up_dict[1]["Vms"],
                       pla_char_up_dict[1]["beta"],
                       pla_char_up_dict[1]["Dp"]]

        errors = error_analysis(mag_up, mag_up_std, mag_down, mag_down_std,
                                pla_up,
                                pla_up_std, pla_down, pla_down_std, char_up,
                                char_std_up, Te_std, doo_Cs_doo_Te,
                                doo_Vms_doo_Te,
                                doo_beta_doo_Te, V_shock, M_A_up, Mms_up,
                                normal,
                                normal_method_id, shock_type)
    # --------------------------------------------------------------------------
    # Drawing a PS plot in a file (named, e.g., '19980205_2105.ps')
    # --------------------------------------------------------------------------

    # jumping to here if jump_to_plotting was set to true
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug',
                   'Sept', 'Oct', 'Nov', 'Dec']
    SC_names = ['ACE', 'Wind', 'STEREOA', 'STEREOB', 'HeliosA', 'HeliosB',
                'Ulysses', 'Cluster3', 'Cluster1', 'Cluster4', 'OMNI',
                'Voyager1', 'Voyager2', 'DSCOVR', 'PSP', 'SolO']
    SC_titles = ['ACE', 'Wind', 'STEREO A', 'STEREO B', 'Helios A', 'Helios B',
                 'Ulysses', 'Cluster 3', 'Cluster 1', 'Cluster 4', 'OMNI',
                 'Voyager 1', 'Voyager 2', 'DSCOVR', 'PSP', 'SolO']
    shock_time = datetime.fromtimestamp(shock_timestamps[i])

    mag_res_format = '(F9.1)'
    if sc in [2, 3, 11, 12]:
        mag_res_format = '(F9.3)'

    # Scale the units of temperature from Kelvins to 10^4 Kelvins
    Tp_mean_up *= 1e-4
    Tp_mean_down *= 1e-4
    errors[21] *= 1e-4
    errors[22] *= 1e-4

    # Output formats
    scalar_ft = '(F9.4)'
    vector_ft = '(F9.4,1X,F9.4,1X,F9.4)'
    V_sc_fmt = '(F10.4)'
    V_vec_fmt = '(F10.4)'

    # If there is no velocity vector data upstream and downstream
    # velocity vectors are written as "-1000000. -1000000. -1000000."
    if bad_vel == 1:
        V_vector_up = [-1e6, -1e6, -1e6]
        V_vector_down = [-1e6, -1e6, -1e6]
        errors[10:13] = [0, 0, 0]
        errors[14:17] = [0, 0, 0]
        V_vec_fmt = '(F10.0)'

    # Shock type to letters:
    shock_type = 'N'

    if type == 1:
        shock_type = 'FF'
    elif type == 2:
        shock_type = 'FR'

    # Initialize global variables for storing the vertical line and last click time
    last_click_time = shock_time  # Start with shock time

    # if sc in [11, 12]:  # Handling datagaps for Voyager spacecraft
    #    Ball = B[np.isfinite(B)]
    #    t_mag_fix = t_mag[np.isfinite(B)]

    # Creating the filename for the plot
    apu = shock_time.strftime('%Y%m%d_%H%M%S')
    fname = f"{apu}_{SC_names[sc]}"

    # Convert the EPOCH columns to datetime objects
    mag_dataframe['EPOCH'] = [datetime.fromtimestamp(element) for element in
                              mag_dataframe["EPOCH"]]
    pla_dataframe['EPOCH'] = [datetime.fromtimestamp(element) for element in
                              pla_dataframe["EPOCH"]]

    # Adjust time range based on sc
    if sc in [4, 5, 6, 11, 12]:
        time_range = [shock_time - timedelta(hours=1),
                      shock_time + timedelta(hours=1)]
    else:
        time_range = [shock_time - timedelta(minutes=30),
                      shock_time + timedelta(minutes=30)]

    # Start of your plotting code
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    # Example plot, replace with actual data and plots
    axs[0].plot(mag_dataframe['EPOCH'].values, mag_dataframe['B'].values,
                label='B [nT]', color='black', linewidth=0.6, markersize=9)
    axs[0].set_ylabel('B [nT]', fontsize=12)
    axs[0].set_xlim(time_range)
    axs[0].tick_params(axis='both', which='both', direction='in', bottom=True,
                       top=True, left=True, right=True, labelsize=12)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[0].tick_params(labelbottom=False)
    title = f"{shock_type} Shock {month_names[shock_time.month - 1]} {shock_time.day}, {shock_time.year}, {shock_time.strftime('%H:%M:%S')} UT, {SC_titles[sc]}"
    axs[0].set_title(title, fontsize=18, weight='light')

    axs[1].plot(pla_dataframe['EPOCH'].values, pla_dataframe['V'].values,
                label='V [km/s]', marker='+', color='black', linewidth=0.6,
                markersize=9)
    axs[1].set_ylabel('V [km/s]', fontsize=12)
    axs[1].set_xlim(time_range)
    axs[1].tick_params(axis='both', which='both', direction='in', bottom=True,
                       top=True, left=True, right=True, labelsize=12)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1].tick_params(labelbottom=False)

    axs[2].plot(pla_dataframe['EPOCH'].values, pla_dataframe['Np'].values,
                label='Np [1/cm^3]', marker='+', color='black', linewidth=0.6,
                markersize=9)
    axs[2].set_ylabel('Np [1/cm^3]', fontsize=12)
    axs[2].set_xlim(time_range)
    axs[2].tick_params(axis='both', which='both', direction='in', bottom=True,
                       top=True, left=True, right=True, labelsize=12)
    axs[2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[2].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[2].tick_params(labelbottom=False)

    axs[3].plot(pla_dataframe['EPOCH'].values, pla_dataframe['Tp'].values,
                label='Tp [K]', marker='+', color='black', linewidth=0.6,
                markersize=9)
    axs[3].set_ylabel('Tp [K]', fontsize=12)
    axs[3].set_xlim(time_range)
    axs[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axs[3].tick_params(axis='both', which='both', direction='in', bottom=True,
                       top=True, left=True, right=True, labelsize=12)
    axs[3].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[3].yaxis.set_minor_locator(AutoMinorLocator(5))

    # Set scientific notation with power limits
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axs[3].yaxis.set_major_formatter(formatter)

    # Calculating and setting the exponent dynamically
    yticks = axs[3].get_yticks()
    max_value = np.max(pla_dataframe['Tp'].dropna().values)

    exponent = int(np.log10(max_value))

    # Constructing formatted y-axis tick labels
    labels = []
    for ytick in yticks:
        mantissa = ytick / 10 ** exponent
        labels.append(f"{mantissa:.1f}$\\times$10$^{{{exponent}}}$")


    # Define the formatter function
    def scientific_formatter(value, tick_number):
        return f"{value / 10 ** exponent:.1f} $\\times$ 10$^{{{exponent}}}$"


    # Set the y-axis major formatter
    axs[3].yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    for ax in axs:
        ax.axvline(shock_time, color='red', linewidth=1)
        ax.axvline(datetime.fromtimestamp(t_down[0]), color='black',
                   linestyle='--', linewidth=1, dashes=[8, 8])
        ax.axvline(datetime.fromtimestamp(t_down[1]), color='black',
                   linestyle='--', linewidth=1, dashes=[8, 8])
        ax.axvline(datetime.fromtimestamp(t_up[0]), color='black',
                   linestyle='--', linewidth=1, dashes=[8, 8])
        ax.axvline(datetime.fromtimestamp(t_up[1]), color='black',
                   linestyle='--', linewidth=1, dashes=[8, 8])

    # Set x-axis formatter for each subplot
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))

    # Adding the text to the bottom left corner
    text_time = 'hhmm'
    text_date = shock_time.strftime('%Y %b %d')
    plt.gcf().text(0.03, 0.06, f'{text_time}\n{text_date}', ha='left',
                   fontsize=13, weight='light')

    # Adding sideways text to the right side of the bottom plot
    side_text = shock_time.strftime('%a %b %d %H:%M:%S %Y')
    axs[3].text(1.005, 0.305, side_text, ha='left', va='center', rotation=90,
                fontsize=6, transform=axs[3].transAxes)

    # Connect the click event to the on_click function
    # Connect the click and key press events to their respective handler functions
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Construct the full file paths
    if unclear:
        fname_ps = os.path.join(unclear_plot_directory, f"{fname}.ps")
        fname_png = os.path.join(unclear_plot_directory, f"{fname}.png")
    else:
        fname_ps = os.path.join(plot_directory, f"{fname}.ps")
        fname_png = os.path.join(plot_directory, f"{fname}.png")

    # Save files
    plt.savefig(fname_png, format="png")
    plt.savefig(fname_ps, format='ps')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if plot_events:
        plt.show()

    # --------------------------------------------------------------------------
    # Write the final shock times to a file
    # --------------------------------------------------------------------------

    t_print = time.strftime("%Y    %#m    %#d    %#H    %#M    %#S",
                            time.localtime(shock_timestamps[i]))
    with open(shock_times_out_fname, 'a') as file3:
        # Write to file (assuming 'filnum1' is the file object)
        if type != 0 and not time_adjusted:
            file3.write(t_print + '\n')

    # --------------------------------------------------------------------------
    # Write the output of the analysis to a file
    # --------------------------------------------------------------------------

    output_line = (
        f"{t_print: >4}     "  # t_print with right alignment, minimum width 4
        f"{shock_type:<3}    "  # shock_type with left alignment, width 3
        f"{sc_pos.iloc[0]:9.4f} {sc_pos.iloc[1]:9.4f} {sc_pos.iloc[2]:9.4f}    "  # sc_pos formatted as vector_ft
        f"{B_mean_up:8.4f} +- {errors[0]:8.4f}    "
        f"{B_vector_up[0]:8.4f} +- {errors[1]:8.4f}    "
        f"{B_vector_up[1]:8.4f} +- {errors[2]:8.4f}    "
        f"{B_vector_up[2]:8.4f} +- {errors[3]:8.4f}    "
        f"{B_mean_down:8.4f} +- {errors[4]:8.4f}    "
        f"{B_vector_down[0]:8.4f} +- {errors[5]:8.4f}    "
        f"{B_vector_down[1]:8.4f} +- {errors[6]:8.4f}    "
        f"{B_vector_down[2]:8.4f} +- {errors[7]:8.4f}    "
        f"{B_ratio:8.4f} +- {errors[8]:8.4f}    "
        f"{V_mean_up:7.4f} +- {errors[9]:7.4f}    "
        f"{V_vector_up[0]:7.4f} +- {errors[10]:7.4f}    "
        f"{V_vector_up[1]:7.4f} +- {errors[11]:7.4f}    "
        f"{V_vector_up[2]:7.4f} +- {errors[12]:7.4f}    "
        f"{V_mean_down:7.4f} +- {errors[13]:7.4f}    "
        f"{V_vector_down[0]:7.4f} +- {errors[14]:7.4f}    "
        f"{V_vector_down[1]:7.4f} +- {errors[15]:7.4f}    "
        f"{V_vector_down[2]:7.4f} +- {errors[16]:7.4f}    "
        f"{V_jump:7.4f} +- {errors[17]:7.4f}    "
        f"{Np_mean_up:7.4f} +- {errors[18]:7.4f}    "
        f"{Np_mean_down:7.4f} +- {errors[19]:7.4f}    "
        f"{Np_ratio:7.4f} +- {errors[20]:7.4f}    "
        f"{Tp_mean_up:7.4f} +- {errors[21]:7.4f}    "
        f"{Tp_mean_down:7.4f} +- {errors[22]:7.4f}    "
        f"{Tp_ratio:7.4f} +- {errors[23]:7.4f}    "
        f"{Cs_mean_up:7.4f} +- {errors[24]:7.4f}    "
        f"{V_A_mean_up:7.4f} +- {errors[25]:7.4f}    "
        f"{Vms_mean_up:7.4f} +- {errors[26]:7.4f}    "
        f"{beta_mean_up:7.4f} +- {errors[27]:7.4f}    "
        f"{normal[0]:7.4f} +- {errors[28]:7.4f}    "
        f"{normal[1]:7.4f} +- {errors[29]:7.4f}    "
        f"{normal[2]:7.4f} +- {errors[30]:7.4f}    "
        f"{shock_theta:7.4f} +- {errors[31]:7.4f}    "
        f"{V_shock:7.4f} +- {errors[32]:7.4f}    "
        f"{M_A_up:7.4f} +- {errors[33]:7.4f}    "
        f"{Mms_up:7.4f} +- {errors[34]:7.4f}    "
        f"{bad_vel: >10}    "  # right-aligned, minimum width 10
        f"{analysis_int_length: >6.1f}    "  # right-aligned, minimum width 6, one digit after decimal
        f"{avg_res_mag:7.3f}    "
        f"{avg_res_pla:5.1f}")

    year, month, day, hour, minute, second = t_print.split()

    output_line_csv = (
        f"{year:>4},"
        f"{month:>2},"
        f"{day:>2},"
        f"{hour:>2},"
        f"{minute:>2},"
        f"{second:>2},"
        f"{shock_type:<2},"
        f"{sc_pos.iloc[0]:.4f},{sc_pos.iloc[1]:.4f},{sc_pos.iloc[2]:.4f},"
        f"{B_mean_up:.4f},{errors[0]:.4f},"
        f"{B_vector_up[0]:.4f},{errors[1]:.4f},"
        f"{B_vector_up[1]:.4f},{errors[2]:.4f},"
        f"{B_vector_up[2]:.4f},{errors[3]:.4f},"
        f"{B_mean_down:.4f},{errors[4]:.4f},"
        f"{B_vector_down[0]:.4f},{errors[5]:.4f},"
        f"{B_vector_down[1]:.4f},{errors[6]:.4f},"
        f"{B_vector_down[2]:.4f},{errors[7]:.4f},"
        f"{B_ratio:.4f},{errors[8]:.4f},"
        f"{V_mean_up:.4f},{errors[9]:.4f},"
        f"{V_vector_up[0]:.4f},{errors[10]:.4f},"
        f"{V_vector_up[1]:.4f},{errors[11]:.4f},"
        f"{V_vector_up[2]:.4f},{errors[12]:.4f},"
        f"{V_mean_down:.4f},{errors[13]:.4f},"
        f"{V_vector_down[0]:.4f},{errors[14]:.4f},"
        f"{V_vector_down[1]:.4f},{errors[15]:.4f},"
        f"{V_vector_down[2]:.4f},{errors[16]:.4f},"
        f"{V_jump:.4f},{errors[17]:.4f},"
        f"{Np_mean_up:.4f},{errors[18]:.4f},"
        f"{Np_mean_down:.4f},{errors[19]:.4f},"
        f"{Np_ratio:.4f},{errors[20]:.4f},"
        f"{Tp_mean_up:.4f},{errors[21]:.4f},"
        f"{Tp_mean_down:.4f},{errors[22]:.4f},"
        f"{Tp_ratio:.4f},{errors[23]:.4f},"
        f"{Cs_mean_up:.4f},{errors[24]:.4f},"
        f"{V_A_mean_up:.4f},{errors[25]:.4f},"
        f"{Vms_mean_up:.4f},{errors[26]:.4f},"
        f"{beta_mean_up:.4f},{errors[27]:.4f},"
        f"{normal[0]:.4f},{errors[28]:.4f},"
        f"{normal[1]:.4f},{errors[29]:.4f},"
        f"{normal[2]:.4f},{errors[30]:.4f},"
        f"{shock_theta:.4f},{errors[31]:.4f},"
        f"{V_shock:.4f},{errors[32]:.4f},"
        f"{M_A_up:.4f},{errors[33]:.4f},"
        f"{Mms_up:.4f},{errors[34]:.4f},"
        f"{bad_vel},"
        f"{analysis_int_length:.1f},"
        f"{avg_res_mag:.3f},"
        f"{avg_res_pla:.1f}"
    )

    with open(analysis_output_fname, 'a') as file2:
        # Write to file (assuming 'filnum2' is the file object)
        file2.write(output_line + '\n')

    if unclear:
        with open(unclear_csv_file, 'a') as file4:
            file4.write(output_line_csv + '\n')
    else:
        with open(csv_file, 'a') as file3:
            file3.write(output_line_csv + '\n')

    # ------------------------------------------------------------------------------
    # At the end of each loop estimate of remaining duration of the analysis is
    # written out
    # ------------------------------------------------------------------------------

    loop_end_time = time.time()
    loop_time = loop_end_time - loop_start_time

    # Convert loop_time to integer seconds
    loop_time_int = int(loop_time)

    # Calculate remaining time in minutes and seconds
    duration_seconds = (N_sh - i - 1) * loop_time_int
    duration_minutes = duration_seconds // 60
    duration_seconds %= 60

    print('This loop took: ' + str(loop_time_int) + " seconds.")
    print('Time left (estimate): ' + str(duration_minutes) + ' minutes ' + str(
        duration_seconds) + ' seconds.')

# print out bad events
if len(bad_events) > 0:

    print(
        'There was not enough data to do proper analysis for the following events:')
    for event in bad_events:
        print(event)