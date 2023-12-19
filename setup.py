# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools

setuptools.setup(
    name="llm-inference",
    version="0.1",
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
)
