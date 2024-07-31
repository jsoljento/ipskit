import datetime

def convert_date_format(date_str):
    # Convert the string into a datetime object
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    
    # Format the datetime object into the desired string format with tabs
    return date_obj.strftime('%Y\t%m\t%d\t%H\t%M\t%S')

def process_dates(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()  # Remove any leading/trailing whitespace characters
            line = line[0:19]
            if line:  # Ensure the line is not empty
                formatted_date = convert_date_format(line)
                outfile.write(formatted_date + '\n')

# Replace 'input.txt' with the path to your input file
# Replace 'output.txt' with the desired path for your output file
input_file = 'unformatted_shock_times.txt'
output_file = 'formatted_shock_times.txt'

process_dates(input_file, output_file)
