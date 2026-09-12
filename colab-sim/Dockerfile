# A deliberately bare stand-in for a fresh Google Colab runtime.
#
# This does NOT pre-install pandas/numpy/statsmodels/etc the way Colab
# actually does — that's on purpose. Every lab notebook's own first cell is
# responsible for installing everything it needs (see the "Environment
# setup" cell in each lab). Testing against a BARER environment than Colab
# is the stricter, safer direction: a notebook that self-installs correctly
# here will also work on a Colab runtime that already has some of these
# packages, but the reverse isn't guaranteed.
#
# Build once, then execute notebooks inside it — see run_labs.py.
FROM python:3.12-slim

# build-essential + libs: prophet's cmdstan backend compiles a C++ binary on
# first use; lightgbm's wheel is usually enough on its own but the toolchain
# costs little to have present.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip jupyter nbconvert ipykernel nbformat

WORKDIR /work
