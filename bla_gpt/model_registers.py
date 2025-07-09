from utils import register_model


@register_model
def register_blagpt():
    from bla_gpt import GPT, GPTConfig

    return GPTConfig, GPT


@register_model
def register_best():
    from best_model_config import BestConfig

    from bla_gpt import GPT

    return BestConfig, GPT


@register_model
def register_tokenformer():
    from bla_gpt import GPT, TokenformerConfig

    return TokenformerConfig, GPT


@register_model
def register_ftp():
    from ftp import FTPConfig, FTPModel

    return FTPConfig, FTPModel


@register_model
def register_hourglass():
    from hourglass_transformer import HourglassConfig, HourglassTransformer

    return HourglassConfig, HourglassTransformer


@register_model
def register_hymba():
    from hymba import HymbaConfig, HymbaForBlaGPT

    return HymbaConfig, HymbaForBlaGPT


@register_model
def register_rene():
    from rene import ReneConfig, ReneLMHeadModel

    return ReneConfig, ReneLMHeadModel


@register_model
def register_rwkv7():
    from rwkv7.model import RWKV7Config, RWKV7Model

    return RWKV7Config, RWKV7Model


@register_model
def register_zamba():
    from bla_gpt.zamba2.config import MambaConfig, MambaModel

    return MambaConfig, MambaModel


@register_model
def register_llada():
    from llada import LLaDA, LLaDAConfig

    return LLaDAConfig, LLaDA


@register_model
def register_duo():
    from duo import Duo, DuoConfig

    return DuoConfig, Duo


@register_model
def register_avey():
    from avey import Avey, AveyConfig

    return AveyConfig, Avey


@register_model
def register_hierarchical():
    from aunet import HierarchicalConfig, HierarchicalTransformer

    return HierarchicalConfig, HierarchicalTransformer
