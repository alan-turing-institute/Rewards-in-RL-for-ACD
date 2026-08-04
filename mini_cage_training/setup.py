from setuptools import setup, find_packages

# Install this directory as the 'CybORG_plus_plus' package so that
# the existing imports in mini_CAGE/ and Training/ work unchanged:
#
#   from CybORG_plus_plus.mini_CAGE.minimal import SimplifiedCAGE
#   from CybORG_plus_plus.mini_CAGE.agents import Meander_minimal
#   ...
#
# Install with:  pip install -e ./mini_cage_training

setup(
    name="CybORG_plus_plus",
    version="1.0.0",
    # Map the top-level 'CybORG_plus_plus' package to this directory,
    # so that 'from CybORG_plus_plus.mini_CAGE.minimal import ...' resolves to
    # mini_cage_training/mini_CAGE/minimal.py
    package_dir={"CybORG_plus_plus": "."},
    packages=["CybORG_plus_plus", "CybORG_plus_plus.mini_CAGE"],
)
