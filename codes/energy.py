## energy -- Gaspard Berthelier
# utilities to compute energy consumption

#imports
import requests
from statistics import mean
from datetime import timedelta
from datetime import datetime

import numpy as np

#time
def get_past_time(minutes=5):
    """returns time minutes ago"""
    d1 = datetime.now()
    if minutes:
        d2 = d1 - timedelta(minutes=minutes)
    return d2

def get_time():
    """returns current time"""
    return datetime.now()


#energy
def get_power(node, site, start, stop, metric="bmc_node_power_watt"):
    """returns consumption on grid5000 node site from start to stop"""
    if node and site:
        start = str(start)[0:-10]
        stop = str(stop)[0:-10]
        start = start[0:10]+"T"+start[11:]
        stop = stop[0:10]+"T"+stop[11:]
        url = f"https://api.grid5000.fr/stable/sites/{site}/metrics?metrics={metric}&nodes={node}&start_time={start}&end_time={stop}"
        print("fetching energy from url : ")
        print(url)
        data = requests.get(url, verify=False).json()
        data = [data[k]["value"] for k in range(len(data))]
        return data
    else:
        return [0]

def get_power_info(node, site, start, stop, metric="bmc_node_power_watt"):
    """returns mean and total consumption"""
    data = get_power(node, site, start, stop, metric="bmc_node_power_watt")
    return round(np.mean(data),4),round(np.sum(data)/1000,4)

