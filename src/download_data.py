"""Fetch the raw data file named by a config.

Two acquisition modes are supported:

``url`` (default)
    Plain HTTP download, used for the Kenneth French industry file and the
    Rdatasets mirror of ``EuStockMarkets``.

``package``
    Materialise a dataset bundled inside a Python package to CSV, so the rest
    of the pipeline reads a plain file and carries no runtime dependency on the
    provider. Currently used for the twenty-stock S&P 500 daily price panel
    shipped with ``skfolio`` (install with ``pip install skfolio``).
"""
from pathlib import Path
import argparse

import yaml


def _from_url(cfg, dest: Path) -> None:
    import requests

    url = cfg['data']['url']
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f'Downloaded {len(r.content):,} bytes from {url} to {dest}')


def _from_package(cfg, dest: Path) -> None:
    name = cfg['data']['package_dataset']
    if name == 'skfolio.sp500':
        try:
            from skfolio.datasets import load_sp500_dataset
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                'This config needs the optional dependency skfolio '
                '(pip install skfolio).'
            ) from exc
        prices = load_sp500_dataset()
    else:
        raise ValueError(f'Unknown packaged dataset: {name}')
    prices.to_csv(dest)
    print(f'Wrote {prices.shape[0]:,} rows x {prices.shape[1]} columns to {dest}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    dest = Path(cfg['data']['raw_file'])
    dest.parent.mkdir(parents=True, exist_ok=True)

    source = cfg['data'].get('source', 'url')
    if source == 'url':
        _from_url(cfg, dest)
    elif source == 'package':
        _from_package(cfg, dest)
    else:
        raise ValueError(f"Unknown data.source '{source}'; expected 'url' or 'package'.")


if __name__ == '__main__':
    main()
