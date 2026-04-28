try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError("Install taltools[data] to use configurations") from e

def save_config(config, out):
    with open(out.replace('\\', '/'), 'w') as fp:
        OmegaConf.save(config=config, f=fp.name)


def load_config(file, resolve=True):
    with open(file.replace('\\', '/'), 'r') as fp:
        cfg = OmegaConf.load(fp.name)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=resolve))
        return cfg