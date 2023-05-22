# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from forecaster.common.recursive_dict import RecursiveDict


def test_recursive_dict():
    a = RecursiveDict({
        "a": 1,
        "b": {
            "b.a": 2,
            "b.b": {
                "b.b.a" : 3,
                "b.b.b" : 4,
            }
        },
        "c": {
            "c.a": 5,
            "c.b": 6,
        },
    })

    b = {
        "a": {
            "a.a": 1,
            "a.b": 2,
        },
        "b": {
            "b.a": 3,
            "b.b": {
                "b.b.a": 1,
                "b.b.c": 5,
            },
            "b.c": 6
        },
        "c": {
            "c.a": {
                "c.a.a": 1,
                "c.a.b": 5,
            },
            "c.c": {
                "c.c.a": 1,
                "c.c.b": 2,
            }
        },
        "d": "new domain"
    }
    assert a["a"] == 1
    a.update(b)
    assert isinstance(a["a"], RecursiveDict)
    assert a["a"]["a.b"] == 2
    assert a["b"]["b.b"]["b.b.a"] == 1
    assert a["b"]["b.b"]["b.b.b"] == 4
    assert a["b"]["b.b"]["b.b.c"] == 5
    assert isinstance(a["c"]["c.a"], RecursiveDict)
    assert a["c"]["c.c"]["c.c.b"] == 2
    assert a["d"] == "new domain"


if __name__ == "__main__":
    test_recursive_dict()
