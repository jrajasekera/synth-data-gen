"""Plugin loading and registration utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict

from . import plugins


class PluginRegistry:
    """Registry for generator and validator plugins."""

    def __init__(self) -> None:
        self.generators: Dict[str, Callable[..., Any]] = {}
        self.validators: Dict[str, Callable[..., Any]] = {}

    def register_generator(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a generator factory under a name."""

        self.generators[name] = factory

    def register_validator(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a validator factory under a name."""

        self.validators[name] = factory

    def load_entrypoint(self, dotted_path: str) -> Callable[..., Any]:
        """Dynamically load a callable via dotted path."""

        module_name, _, attr = dotted_path.rpartition(".")
        module = import_module(module_name)
        return getattr(module, attr)


registry = PluginRegistry()

__all__ = ["registry", "PluginRegistry", "plugins"]
