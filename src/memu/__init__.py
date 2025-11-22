try:
    from memu._core import hello_from_bin
except Exception:
    def hello_from_bin() -> str:
        return "memu"


def _rust_entry() -> str:
    return hello_from_bin()
