#!/bin/sh
# Create the virtualenv and, if the machine has only a JRE, fetch a JDK into .venv/jdk.
# Terrier is a Java engine and pyjnius needs javac, so a plain java is not enough.
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install python-terrier ir_measures pandas install-jdk weasyprint

if ! command -v javac >/dev/null 2>&1 && [ ! -x "$JAVA_HOME/bin/javac" ]; then
    echo "No system JDK found; downloading Temurin 17 into .venv/jdk ..."
    .venv/bin/python -c "import jdk, os; print(jdk.install('17', jre=False, path=os.path.abspath('.venv/jdk')))"
fi

tar xzf cran.tar.gz          # cran.all.1400, cran.qry, cranqrel, cranqrel.readme

echo
echo "Done. Next:"
echo "  .venv/bin/python search_ninjas_experiments.py    # the experiment plan"
echo "  .venv/bin/python search_ninjas_search.py --evaluate"
echo "  .venv/bin/python search_ninjas_report.py         # the submission PDF"
