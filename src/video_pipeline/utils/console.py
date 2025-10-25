ï»¿# -*- coding: utf-8 -*-
def safe_print(*args, **kwargs):
    """
    Ã‰crit sur stdout. Si stdout est fermÃ© (pytest capture/flush),
    bascule sur sys.__stdout__. Ne lÃ¨ve jamais dâ€™exception.
    """
    import sys
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            stream = kwargs.get("file", None)
            if stream is None or getattr(stream, "closed", False):
                stream = getattr(sys, "__stdout__", None) or getattr(sys, "stdout", None)
            if stream is not None:
                text = " ".join(str(a) for a in args)
                if text and not text.endswith("\n"):
                    text += "\n"
                stream.write(text)
                try: stream.flush()
                except Exception: pass
        except Exception:
            pass


