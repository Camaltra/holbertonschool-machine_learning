#!/usr/bin/env python3


"""
Useless
"""


import requests

UPCOMING_URL = "https://api.spacexdata.com/v4/launches/upcoming"
BASE_URL_ROCKET = "https://api.spacexdata.com/v4/rockets/"
BASE_URL_LAUNCHPAD = "https://api.spacexdata.com/v4/launchpads/"


def get_next_launch(api_response):
    """
    Holberton is a mess
    :param api_response:
    :return:
    """
    dates = [x["date_unix"] for x in api_response]
    index = dates.index(min(dates))
    return response[index]


if __name__ == "__main__":
    response = requests.get(UPCOMING_URL).json()

    next_launch = get_next_launch(response)

    name = next_launch.get("name")
    date = next_launch.get("date_local")
    rocket_id = next_launch.get("rocket")
    launchpad_id = next_launch.get("launchpad")

    response_rocket = requests.get(BASE_URL_ROCKET + rocket_id).json()
    rocket_name = response_rocket["name"]

    response_launchpad = requests.get(BASE_URL_LAUNCHPAD + launchpad_id).json()
    launchpad_name = response_launchpad["name"]
    launchpad_loc = response_launchpad["locality"]

    print(
        name
        + " ("
        + date
        + ") "
        + rocket_name
        + " - "
        + launchpad_name
        + " ("
        + launchpad_loc
        + ")"
    )
