from .flash_topk import FlashTopK
__all__ = [
    'FlashTopK',
]

# if there's time/interest maybe also do a triton/cutlass(?) parallel kernel here idk how much overhead the flex block thing has