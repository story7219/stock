import os
import pickle
import hashlib
import time
from functools import lru_cache
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str = './.cache', memory_size: int = 128, expire: int = 3600):
        self.cache_dir = cache_dir
        self.memory_size = memory_size
        self.expire = expire
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}
        self._memory_time = {}

    def _make_key(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, *args, **kwargs) -> Optional[Any]:
        key = self._make_key(*args, **kwargs)
        # 1. 메모리 캐시
        if key in self._memory_cache:
            if time.time() - self._memory_time[key] < self.expire:
                return self._memory_cache[key]
            else:
                del self._memory_cache[key]
                del self._memory_time[key]
        # 2. 디스크 캐시
        path = os.path.join(self.cache_dir, key + '.pkl')
        if os.path.exists(path):
            if time.time() - os.path.getmtime(path) < self.expire:
                with open(path, 'rb') as f:
                    value = pickle.load(f)
                    self._memory_cache[key] = value
                    self._memory_time[key] = time.time()
                    return value
            else:
                os.remove(path)
        return None

    def set(self, value: Any, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        # 1. 메모리 캐시
        self._memory_cache[key] = value
        self._memory_time[key] = time.time()
        # 2. 디스크 캐시
        path = os.path.join(self.cache_dir, key + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(value, f)

    def clear(self):
        self._memory_cache.clear()
        self._memory_time.clear()
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, f)) 