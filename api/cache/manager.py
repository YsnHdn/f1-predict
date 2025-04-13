"""
Cache manager for API clients.
This module provides caching functionality to reduce repeated API calls and improve performance.
"""

import os
import json
import pickle
import logging
import hashlib
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages cache for API clients to improve performance and reduce API calls.
    Supports both memory cache and disk cache with configurable expiration times.
    """
    
    def __init__(self, namespace: str, cache_dir: Optional[str] = None, 
                 memory_cache: bool = True, disk_cache: bool = True,
                 default_expiry: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            namespace: Namespace for this cache to avoid conflicts
            cache_dir: Directory to store cache files (default: ./api/cache/data/)
            memory_cache: Whether to use in-memory caching
            disk_cache: Whether to use disk caching
            default_expiry: Default cache expiry time in seconds (86400 = 1 day)
        """
        self.namespace = namespace
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache
        self.default_expiry = default_expiry
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("api/cache/data") / namespace
            
        if self.disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize in-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Cache manager initialized for namespace '{namespace}' with "
                   f"memory_cache={memory_cache}, disk_cache={disk_cache}")
    
    def _get_cache_key(self, key: str) -> str:
        """
        Generate a standardized cache key with namespace.
        
        Args:
            key: Original key to format
        
        Returns:
            Formatted cache key
        """
        # Generate MD5 hash of the key for filesystem safety
        if isinstance(key, str):
            hash_key = hashlib.md5(key.encode()).hexdigest()
        else:
            hash_key = hashlib.md5(str(key).encode()).hexdigest()
        return f"{self.namespace}_{hash_key}"
    
    def _get_cache_file_path(self, key: str) -> Path:
        """
        Get file path for a cache key.
        
        Args:
            key: Cache key
        
        Returns:
            Path object for cache file
        """
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry: Optional expiry time in seconds (overrides default)
        """
        if expiry is None:
            expiry = self.default_expiry
            
        # Calculate expiry timestamp
        expiry_time = datetime.now() + timedelta(seconds=expiry)
        cache_entry = {
            'value': value,
            'expiry': expiry_time
        }
        
        cache_key = self._get_cache_key(key)
        
        # Set in memory cache
        if self.memory_cache:
            self._cache[cache_key] = cache_entry
            
        # Set in disk cache
        if self.disk_cache:
            try:
                cache_file = self._get_cache_file_path(key)
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_entry, f)
            except Exception as e:
                logger.error(f"Error writing to disk cache: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found or expired
        """
        cache_key = self._get_cache_key(key)
        value = None
        
        # Try memory cache first
        if self.memory_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if cache_entry['expiry'] > datetime.now():
                logger.debug(f"Memory cache hit for {key}")
                return cache_entry['value']
            else:
                # Expired entry, remove from memory
                logger.debug(f"Memory cache expired for {key}")
                del self._cache[cache_key]
        
        # Try disk cache if memory cache missed
        if self.disk_cache:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_entry = pickle.load(f)
                        
                    if cache_entry['expiry'] > datetime.now():
                        logger.debug(f"Disk cache hit for {key}")
                        # Update memory cache
                        if self.memory_cache:
                            self._cache[cache_key] = cache_entry
                        return cache_entry['value']
                    else:
                        # Expired entry, remove file
                        logger.debug(f"Disk cache expired for {key}")
                        cache_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error reading from disk cache: {str(e)}")
        
        logger.debug(f"Cache miss for {key}")
        return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if the key was found and deleted, False otherwise
        """
        cache_key = self._get_cache_key(key)
        found = False
        
        # Remove from memory cache
        if self.memory_cache and cache_key in self._cache:
            del self._cache[cache_key]
            found = True
            
        # Remove from disk cache
        if self.disk_cache:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    found = True
                except Exception as e:
                    logger.error(f"Error deleting cache file: {str(e)}")
        
        return found
    
    def clear(self) -> None:
        """Clear all cache entries for this namespace."""
        # Clear memory cache
        if self.memory_cache:
            self._cache.clear()
            
        # Clear disk cache
        if self.disk_cache:
            try:
                for cache_file in self.cache_dir.glob('*.pkl'):
                    if cache_file.name.startswith(f"{self.namespace}_"):
                        cache_file.unlink()
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")
        
        logger.info(f"Cache cleared for namespace '{self.namespace}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'namespace': self.namespace,
            'memory_cache_enabled': self.memory_cache,
            'disk_cache_enabled': self.disk_cache,
            'default_expiry_seconds': self.default_expiry,
            'memory_cache_entries': 0,
            'disk_cache_entries': 0,
            'memory_cache_size_bytes': 0,
            'disk_cache_size_bytes': 0
        }
        
        # Memory cache stats
        if self.memory_cache:
            stats['memory_cache_entries'] = len(self._cache)
            
        # Disk cache stats
        if self.disk_cache:
            try:
                disk_files = list(self.cache_dir.glob(f"{self.namespace}_*.pkl"))
                stats['disk_cache_entries'] = len(disk_files)
                stats['disk_cache_size_bytes'] = sum(f.stat().st_size for f in disk_files)
            except Exception as e:
                logger.error(f"Error calculating disk cache stats: {str(e)}")
                
        return stats