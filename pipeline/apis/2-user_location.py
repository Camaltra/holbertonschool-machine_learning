#!/usr/bin/env python3


"""
useless
"""


import requests
import sys
import time


DEFAULT_HEADERS = {"Accept": "application/vnd.github.v3+json"}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit()
    url = sys.argv[1]
    r = requests.get(url, DEFAULT_HEADERS)

    if r.status_code == 200:
        print(r.json()["location"])

    elif r.status_code == 404:
        print("Not found")

    elif r.status_code == 403:
        rate_limit = int(r.headers["X-Ratelimit-Reset"])
        now = int(time.time())
        minutes = int((rate_limit - now) / 60)
        print("Reset in {} min".format(minutes))
