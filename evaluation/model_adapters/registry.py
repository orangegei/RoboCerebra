from model_adapters.openvla import OpenVLAAdapter
from model_adapters.pi0 import Pi0Adapter


_ADAPTERS = {
    "openvla": OpenVLAAdapter(),
    "pi0": Pi0Adapter(),
}


def get_policy_adapter(model_family: str):
    try:
        return _ADAPTERS[model_family]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported model family: {model_family}. Supported: {supported}") from exc
