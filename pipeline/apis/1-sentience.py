#!/usr/bin/env python3

"""Useless comment"""

import requests


def sentientPlanets():
    """
    Useless comment
    """
    url = "https://swapi-api.hbtn.io/api/species"
    sentient_planets = []
    while url is not None:
        data = requests.get(url).json()
        species = data.get("results")
        for specie in species:
            if specie.get('designation') == 'sentient' or \
                    specie.get('classification') == 'sentient':
                homeworld_url = specie.get("homeworld")

                if homeworld_url is None:
                    continue
                sentient_planets.append(
                    requests.get(homeworld_url).json().get("name")
                )

        url = data.get("next")

    return sentient_planets
