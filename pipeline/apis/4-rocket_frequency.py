#!/usr/bin/env python3


"""
Useless
"""


import requests


BASE_LAUNCH_URL = 'https://api.spacexdata.com/v4/launches'
BASE_ROCKET_URL = "https://api.spacexdata.com/v4/rockets/"

if __name__ == '__main__':

    rockets = {}

    launches = requests.get(BASE_LAUNCH_URL).json()

    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket = requests.get(BASE_ROCKET_URL + rocket_id).json()
        rocket_name = rocket.get('name')

        if rocket_name in rockets.keys():
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1

    sort = sorted(rockets.items(), key=lambda x: x[0])
    sort = sorted(sort, key=lambda x: x[1], reverse=True)

    for i in sort:
        print("{}: {}".format(i[0], i[1]))
