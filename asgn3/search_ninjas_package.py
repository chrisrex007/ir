"""Package only the files required to inspect and grade the assignment."""
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parent

# Source, dependencies, local training data, trained hidden-query model,
# held-out test results, and the report. Internal diagnostics stay in Git.
SUBMISSION_FILES = (
    'README.md',
    'requirements.txt',
    'cran.all.1400',
    'cran.qry',
    'cranqrel',
    'search_ninjas_group.py',
    'search_ninjas_data.py',
    'search_ninjas_features.py',
    'search_ninjas_rank.py',
    'search_ninjas_report.py',
    'search_ninjas_artifacts/search_ninjas_final_model.joblib',
    'search_ninjas_artifacts/search_ninjas_report.pdf',
    'search_ninjas_artifacts/search_ninjas_metrics.csv',
    'search_ninjas_artifacts/search_ninjas_zone_test_seqid.txt',
    'search_ninjas_artifacts/search_ninjas_svm_test_seqid.txt',
)


def package():
    from search_ninjas_verify import verify

    verify(BASE / 'search_ninjas_artifacts')
    missing = [name for name in SUBMISSION_FILES if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing submission files: {missing}')
    archive_path = BASE / 'search_ninjas_submission.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in SUBMISSION_FILES:
            archive.write(BASE / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != list(SUBMISSION_FILES) or archive.testzip() is not None:
            raise ValueError('Submission archive verification failed')
    print(f'Created {archive_path}: {len(SUBMISSION_FILES)} required files')


if __name__ == '__main__':
    package()
