"""Locate the JDK that PyTerrier needs before ``import pyterrier`` runs.

Terrier is a Java engine, so pyjnius has to find a JVM at import time. This
machine has no system Java; ``search_ninjas_setup.sh`` drops a private JDK under
``.venv/jdk``. Import this module *first*, before any pyterrier import, and it
points JAVA_HOME at that JDK (leaving an existing JAVA_HOME alone).
"""

import glob
import os
import sys

VENV_JDK_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "jdk", "jdk-*")


def ensure_java_home():
    """Set JAVA_HOME/PATH to the bundled JDK unless the caller already has one."""
    if os.environ.get("JAVA_HOME") and os.path.exists(os.path.join(os.environ["JAVA_HOME"], "bin", "javac")):
        return os.environ["JAVA_HOME"]

    candidates = sorted(glob.glob(VENV_JDK_GLOB))
    if not candidates:
        sys.exit(
            "No JDK found. Run ./search_ninjas_setup.sh to create .venv and download one,\n"
            "or set JAVA_HOME to an existing JDK 11+ installation."
        )

    java_home = candidates[-1]
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    return java_home


ensure_java_home()
