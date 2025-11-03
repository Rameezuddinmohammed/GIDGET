"""Singleton metaclass for shared instances."""

import threading
from typing import Any, Dict, Type


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    
    _instances: Dict[Type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """Create or return existing instance."""
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all singleton instances (for testing)."""
        with cls._lock:
            cls._instances.clear()