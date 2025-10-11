"""Built-in generator plugins."""

from collections.abc import Callable
from typing import Any

GeneratorFactory = Callable[[dict[str, Any]], Callable[[dict[str, Any]], Any]]

__all__ = ["GeneratorFactory"]
