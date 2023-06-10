#!/bin/bash

python create_table.py
exec /entrypoint "${@}"
