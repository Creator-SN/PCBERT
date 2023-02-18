from __future__ import annotations
import os
import pickle
from typing import Any


class FileCache():

    def __init__(self,dir="./temp",debug=True) -> None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.root = dir
        self.debug = debug

    def exists(self,key:str)->bool:
        path = self._get_path(f"{key}")
        return os.path.exists(path)

    def save(self,key:str,obj:Any,overwrite:bool=False)->bool:
        path = self._get_path(f"{key}")
        if self.exists(key) and not overwrite:
            return False
        with open(path,"wb") as f:
            pickle.dump(obj,f)
        return True

    def load(self,key,construct=None)->Any:
        path = self._get_path(f"{key}")
        if self.exists(key):
            with open(path,"rb") as f:
                if self.debug:
                    print(f"load cached {path}")
                return pickle.load(f)
        if construct is not None:
            obj = construct()
            self.save(key,obj)
            return obj
        return None

    def group(self,key)->FileCache:
        path = self._get_path(f"{key}")
        return FileCache(path)
        
    def _get_path(self,key:str)->str:
        return os.path.join(self.root,key)
