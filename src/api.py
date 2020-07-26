import json
import requests
import pandas as pd


def get_us_daily():
    # API Call and data preparation
    headers = {}
    payload = {}

    url = "https://covidtracking.com/api/us/daily"
    response_US_Daily = requests.request("GET", url, headers=headers, data=payload)
    US_daily = pd.DataFrame(json.loads(response_US_Daily.text))
    US_daily['date'] = pd.to_datetime(US_daily['date'], format='%Y%m%d')
    US_daily.set_index('date', inplace=True)
    return US_daily

def get_states_daily():
    # API Call and data preparation
    headers = {}
    payload = {}

    url = "https://covidtracking.com/api/states/daily"
    response_US_States = requests.request("GET", url, headers=headers, data=payload)
    US_states = pd.DataFrame(json.loads(response_US_States.text))
    US_states['date'] = pd.to_datetime(US_states['date'], format='%Y%m%d')
    # US_states.set_index('date', inplace=True)
    return US_states
