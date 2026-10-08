"""Audit trail written next to every run (phase 0, A6)."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path


def _sha256(path: Path, strip_cr: bool = False) -> str:
    b = path.read_bytes()
    if strip_cr:  # CSV checked out with CRLF on Windows hashes like the LF file
        b = b.replace(b'\r\n', b'\n')
    return hashlib.sha256(b).hexdigest()


def audit_trail(cfg: dict) -> dict:
    """Commit, input-data hash and package versions of the current run."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'scripts',
                                     'config'], capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    raw = Path(cfg['data'].get('raw_file', ''))
    data_hash = _sha256(raw, strip_cr=raw.suffix == '.csv') if raw.is_file() else None
    versions = {'python': platform.python_version()}
    for mod in ('numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'skfolio'):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {'code_commit': commit, 'code_dirty': dirty, 'data_file': str(raw),
            'data_sha256': data_hash, 'versions': versions}
