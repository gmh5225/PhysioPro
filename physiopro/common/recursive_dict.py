# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import collections.abc
from typing import Iterable, Tuple, overload, Mapping, Any, Union, Optional


class RecursiveDict(dict):
    def __init__(self, d: Optional[Union[Mapping[Any, Any], Iterable[Tuple[Any, Any]]]] = None, **kwargs):
        super().__init__()
        if d is not None:
            if "keys" in dir(d):
                for k in d.keys():
                    v = d[k]
                    if isinstance(v, collections.abc.Mapping):
                        v = RecursiveDict(v)
                    self[k] = v
            else:
                for k, v in d:
                    if isinstance(v, collections.abc.Mapping):
                        v = RecursiveDict(v)
                    self[k] = v
        for k in kwargs:
            v = d[k]
            if isinstance(v, collections.abc.Mapping):
                v = RecursiveDict(v)
            self[k] = v

    def update(self, E: Optional[Mapping[Any, Any]] = None, **F):
        if E is not None:
            if "keys" in dir(E):
                for k in E:
                    if k not in self:
                        self[k] = E[k]
                    elif isinstance(self[k], collections.abc.Mapping) and isinstance(E[k], collections.abc.Mapping):
                        self[k].update(E[k])
                    else:
                        self[k] = E[k]
            else:
                for k, v in E:
                    if k not in self:
                        self[k] = v
                    elif isinstance(self[k], collections.abc.Mapping) and isinstance(v, collections.abc.Mapping):
                        self[k].update(v)
                    else:
                        self[k] = E[k]
        for k in F:
            if k not in self:
                self[k] = F[k]
            elif isinstance(self[k], collections.abc.Mapping) and isinstance(F[k], collections.abc.Mapping):
                self[k].update(F[k])
            else:
                self[k] = F[k]

    def __setitem__(self, k, v):
        if isinstance(v, collections.abc.Mapping) and not isinstance(v, RecursiveDict):
            super().__setitem__(k, RecursiveDict(v))
        else:
            super().__setitem__(k, v)

    def __str__(self) -> str:
        return super().__str__()

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def get(self, key: Any, default: Any = None):
        if key in self:
            return self[key]
        else:
            return default
