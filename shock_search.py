from datetime import datetime

from ipsvm import IPSVM
from ipsvm.config import Config

config = Config()
config.update({'CACHE': './cache'})

ipsvm = IPSVM(config)

spacecraft = 'ACE'
start_time = datetime(2013, 6, 1)
end_time = datetime(2013, 9, 1)

shocks = ipsvm.bigscan(spacecraft, start_time, end_time)

shock_candidates_file = './shock_time_formatting/unformatted_shock_times.txt'

with open(shock_candidates_file, 'w') as file:
    for i in range(len(shocks)):
        file.write(str(shocks[i][0]) + " " + str(shocks[i][1]) +  '\n')
