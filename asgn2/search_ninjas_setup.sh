#!/bin/sh
# Create the virtual environment this assignment runs in.
#
# PyTerrier drives Terrier, which is a Java engine, so a JDK is needed as well
# as the Python packages. If the machine has no system Java, one is downloaded
# into .venv/jdk and search_ninjas_env.py picks it up from there -- nothing is
# installed outside this directory.
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install python-terrier ir_measures pandas install-jdk weasyprint

if ! command -v javac >/dev/null 2>&1 && [ -z "$JAVA_HOME" ]; then
    echo "No system JDK found; downloading Temurin 17 into .venv/jdk ..."
    .venv/bin/python -c "import jdk, os; print(jdk.install('17', jre=False, path=os.path.abspath('.venv/jdk')))"
fi

tar xzf cran.tar.gz          # cran.all.1400, cran.qry, cranqrel, cranqrel.readme

echo
echo "Done. Next:"
echo "  .venv/bin/python search_ninjas_experiments.py   # the full experiment plan"
echo "  .venv/bin/python search_ninjas_search.py --evaluate"
echo "  .venv/bin/python search_ninjas_report.py         # the submission PDF"
