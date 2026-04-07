#!/usr/bin/env python3
"""Helper: Get ngrok tunnel URL."""
import json, sys, urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:4040/api/tunnels")
    data = json.loads(resp.read())
    url = data["tunnels"][0]["public_url"]
    print(url)
except Exception:
    sys.exit(1)
