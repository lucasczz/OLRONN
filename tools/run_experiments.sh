#!/bin/bash

python3 tools/schedule.py ; python3 tools/optimizers.py ; python3 tools/pretune.py ; python3 tools/model_sizes.py

