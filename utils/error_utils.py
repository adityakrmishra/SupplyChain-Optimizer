from typing import Dict, Any
import logging
import sys
from functools import wraps

class ErrorHandler:
    """Centralized error handling utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger('error_handler')
        
    def handle(self, exceptions: tuple, message: str = None):
        """Decorator for exception handling"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    self.log.error(f"{message or 'Error'}: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def log_exception(self, error: Exception, context: Dict[str, Any]):
        """Log detailed error information"""
        self.logger.error(
            "Error occurred",
            extra={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'system_info': {
                    'platform': sys.platform,
                    'python_version': sys.version
                }
            }
        )
