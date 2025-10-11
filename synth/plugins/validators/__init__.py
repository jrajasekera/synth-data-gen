"""Built-in validator plugins."""

from collections.abc import Callable
from typing import Any

ValidatorFactory = Callable[[dict[str, Any]], Callable[[dict[str, Any]], None]]

__all__ = ["ValidatorFactory"]
