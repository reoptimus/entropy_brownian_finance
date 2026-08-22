from pathlib import Path
import requests
import yaml


def main():
    cfg = yaml.safe_load(Path('config/default.yaml').read_text())
    url = cfg['data']['url']
    dest = Path(cfg['data']['raw_zip'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f'Downloaded {len(r.content):,} bytes to {dest}')


if __name__ == '__main__':
    main()
