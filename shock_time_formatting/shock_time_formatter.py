import datetime

def convert_date_format(input_date_str):
    """Convert a datetime string to have the correct formatting.

    This function converts a datetime string given by the IPSVM
    algorithm to have the correct formatting so that it can be read
    by the shock analysis program.

    Parameters
    ----------
    input_date_str : str
        Input datetime string.

    Returns
    -------
    output_date_str : str
        Output datetime string with the correct formatting.
    """

    date_obj = datetime.datetime.strptime(input_date_str, '%Y-%m-%d %H:%M:%S')
    
    # Format the datetime object into the desired string format with
    # four spaces between each part
    output_date_str = date_obj.strftime('%Y    %m    %d    %H    %M    %S')

    return output_date_str

def process_dates(input_file, output_file):
    """Convert datetimes to the correct format and save the results.

    This function takes the datetime strings given in the input file,
    reformats them to have the correct format, and saves the resulting
    datetime strings to the output file.

    Parameters
    ----------
    input_file : str
        Input file path.
    output_file : str
        Output file path.
    """

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Remove any leading/trailing whitespace characters
            line = line.strip()
            line = line[0:19]
            if line:  # Ensure the line is not empty
                formatted_date = convert_date_format(line)
                outfile.write(formatted_date + '\n')


input_file = 'unformatted_shock_times.txt'
output_file = 'formatted_shock_times.txt'

process_dates(input_file, output_file)
