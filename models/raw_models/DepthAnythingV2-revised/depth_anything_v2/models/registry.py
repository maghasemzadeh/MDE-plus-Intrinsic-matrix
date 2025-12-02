"""
Model registry for managing and loading different depth estimation models.
"""
from typing import Dict, Type, Optional, List
from .base import BaseDepthModel


class ModelRegistry:
    """
    Registry for depth estimation models.
    Allows registration and retrieval of different model types.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, Type[BaseDepthModel]] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(
        self, 
        name: str, 
        model_class: Type[BaseDepthModel],
        alias: Optional[str] = None
    ) -> None:
        """
        Register a model class.
        
        Args:
            name: Unique name for the model
            model_class: Model class that inherits from BaseDepthModel
            alias: Optional alias name for the model (if different from name)
        """
        if not issubclass(model_class, BaseDepthModel):
            raise TypeError(f"Model class must inherit from BaseDepthModel, got {type(model_class)}")
        
        self._models[name] = model_class
        
        if alias and alias != name:
            self._aliases[alias] = name
    
    def get(self, name: str) -> Optional[Type[BaseDepthModel]]:
        """
        Get a model class by name.
        
        Args:
            name: Model name or alias
        
        Returns:
            Model class if found, None otherwise
        """
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]
        
        return self._models.get(name)
    
    def list(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def has(self, name: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            name: Model name or alias
        
        Returns:
            True if model is registered, False otherwise
        """
        if name in self._aliases:
            return True
        return name in self._models


# Global registry instance
_global_registry = ModelRegistry()


def register_model(
    name: str,
    model_class: Type[BaseDepthModel],
    alias: Optional[str] = None
) -> None:
    """
    Register a model in the global registry.
    
    Args:
        name: Unique name for the model
        model_class: Model class that inherits from BaseDepthModel
        alias: Optional alias name for the model
    """
    _global_registry.register(name, model_class, alias)


def get_model(name: str) -> Optional[Type[BaseDepthModel]]:
    """
    Get a model class from the global registry.
    
    Args:
        name: Model name or alias
    
    Returns:
        Model class if found, None otherwise
    """
    return _global_registry.get(name)


def list_models() -> List[str]:
    """
    List all registered model names in the global registry.
    
    Returns:
        List of model names
    """
    return _global_registry.list()


def has_model(name: str) -> bool:
    """
    Check if a model is registered in the global registry.
    
    Args:
        name: Model name or alias
    
    Returns:
        True if model is registered, False otherwise
    """
    return _global_registry.has(name)


__all__ = [
    'ModelRegistry',
    'register_model',
    'get_model',
    'list_models',
    'has_model',
]

