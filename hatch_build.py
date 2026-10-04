"""Wheel build hook: a wheel that carries the orthant binary gets a platform tag.

The orthant package ships CPython 3.11 and 3.12 x86-64 Linux extensions. A
wheel that includes a binary loadable by the building interpreter must not be
tagged pure Python. When there is none -- another platform, or a build from the sdist,
which carries no binaries -- the wheel stays pure.
"""

import os
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ORTHANT = os.path.join("src", "multivariate_probit", "orthant")


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if os.path.exists(os.path.join(self.root, ORTHANT, "_ofree" + suffix)):
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
