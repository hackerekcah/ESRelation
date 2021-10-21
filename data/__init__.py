DATA_REGISTRY = dict()


def register_dataset(cls):
    DATA_REGISTRY[cls.__name__] = cls
    return cls


from data import dcase18_dataset, esc_dataset, urbansound8k


__all__ = ["register_dataset"]