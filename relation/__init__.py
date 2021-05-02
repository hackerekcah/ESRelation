RBLOCK_REGISTRY = dict()


def register_rblock(name):
    def decorated(cls):
        RBLOCK_REGISTRY[name] = cls
        return cls
    return decorated


from . import rblock, rblock_pe, rblock_efficient, rblock_pe_efficient

__all__ = ["register_rblock"]