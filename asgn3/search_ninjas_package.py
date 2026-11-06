"""Build a submission archive with an integrity manifest, excluding environments."""
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parent


def package():
    from search_ninjas_verify import verify
    verify(BASE / 'search_ninjas_artifacts')
    files = sorted(set(BASE.glob('search_ninjas_*.py')) | set((BASE / 'tests').glob('*.py')) |
                   {BASE/name for name in ('README.md', 'requirements.txt', 'PA3.pdf',
                     'cran.all.1400', 'cran.qry', 'cranqrel', 'cranqrel.readme')} |
                   {p for p in (BASE / 'search_ninjas_artifacts').iterdir() if p.is_file()})
    manifest = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = BASE / 'search_ninjas_submission.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(BASE)))
        archive.writestr('search_ninjas_manifest.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(f'Created {path.name}: {len(files)} files, manifest verified')


if __name__ == '__main__':
    package()
