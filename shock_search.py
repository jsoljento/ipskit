from datetime import datetime

from ipsvm import IPSVM
from ipsvm.config import Config

config = Config()
config.update({'CACHE': './cache'})

ipsvm = IPSVM(config)

spacecraft = 'ACE'
start_time = datetime(2013, 9, 1)
end_time = datetime(2013, 12, 27)

shocks = ipsvm.bigscan(spacecraft, start_time, end_time)
