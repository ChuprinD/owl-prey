from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")


if not AZURE_CONNECTION_STRING:
    raise RuntimeError("Environment variable AZURE_CONNECTION_STRING is not set.")

if not AZURE_CONTAINER_NAME:
    raise RuntimeError("Environment variable AZURE_CONTAINER_NAME is not set.")