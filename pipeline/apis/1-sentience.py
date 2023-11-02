#!/usr/bin/env python3

import requests


def sentientPlanets():
    """
    Useless comment
    """
    data = requests.get("https://swapi-api.hbtn.io/api/species").json()
    sentient_planets = []
    visited_homeworld_url = set()
    while data.get("next") is not None:
        species = data.get("results")
        for specie in species:
            if specie.get('designation') == 'sentient' or \
                    specie.get('classification') == 'sentient':
                homeworld_url = specie.get("homeworld")
                if homeworld_url in visited_homeworld_url \
                        or homeworld_url is None:
                    continue
                sentient_planets.append(
                    requests.get(homeworld_url).json().get("name")
                )
                visited_homeworld_url.add(homeworld_url)

        data = requests.get(data.get("next")).json()

    return sentient_planets
