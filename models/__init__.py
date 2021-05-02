ARCH_REGISTRY = dict()


def register_arch(cls):
    ARCH_REGISTRY[cls.__name__] = cls
    return cls


from . import resnext_baseline, resnext_relation

__all__ = ["register_arch"]