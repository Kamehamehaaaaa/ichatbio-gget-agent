import importlib
from importlib.resources.abc import Traversable


def resource(*path, text=True) -> str | Traversable:
    file = importlib.resources.files("resources").joinpath(*path)
    if text:
        return file.read_text()
    return file
