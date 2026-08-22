from pathlib import Path
import argparse
import requests
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    url = cfg['data']['url']
    dest = Path(cfg['data']['raw_file'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f'Downloaded {len(r.content):,} bytes to {dest}')


if __name__ == '__main__':
    main()
