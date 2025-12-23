import json
import pickle
from pathlib import Path


def init_directories(*dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)


def read_json(file):
    try:
        with open(file, 'rb') as j:
            return json.loads(j.read())
    except Exception as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_json(j, dst):
    with open(dst, 'w') as f:
        json.dump(j, f)


def read_pkl(file):
    try:
        with open(file, 'rb') as p:
            return pickle.load(p)
    except Exception as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_pkl(p, dst):
    with open(dst, 'wb') as f:
        pickle.dump(p, f)
