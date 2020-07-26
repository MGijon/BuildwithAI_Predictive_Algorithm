import datetime
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

def fill_data(data, end_date):
    start = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    step = datetime.timedelta(days=1)
    states = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN',
          'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
          'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI',
          'WV', 'WY']

    for state in states:
        date = start
        # date_str = date.strftime('%Y-%m-%d')

        index = data[(data['state'] == state) & (data['date'] == date)].index

        while not data.loc[index].empty:
            recovered = data.loc[index, 'recovered']
            if recovered.isnull().bool():
                check_prev(data, state, 'recovered', date, step, index)

            positive = data.loc[index, 'positive']
            if positive.isnull().bool():
                check_prev(data, state, 'positive', date, step, index)

            date -= step
            index = data[(data['state'] == state) & (data['date'] == date)].index

    return data


def check_prev(data, state, column, date, step, index):
    prev_date = date - step
    prev_index = data[(data['state'] == state) & (data['date'] == prev_date.strftime('%Y-%m-%d'))].index
    prev_val = data.loc[prev_index, column]
    if prev_val.empty:
        data.loc[index, column] = 0
        return 0
    if not prev_val.isnull().bool():
        value = prev_val.values[0]
    else:
        value = check_prev(data, state, column, prev_date, step, prev_index)

    data.loc[index, column] = value
    return value
