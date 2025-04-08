
import socket
import pytest
import requests
import sqlite3

try:
    import sqlalchemy
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from codeflash.picklepatch.pickle_patcher import PicklePatcher
from codeflash.picklepatch.pickle_placeholder import PicklePlaceholder
def test_picklepatch_simple_nested():
    """
    Test that a simple nested data structure pickles and unpickles correctly.
    """
    original_data = {
        "numbers": [1, 2, 3],
        "nested_dict": {"key": "value", "another": 42},
    }

    dumped = PicklePatcher.dumps(original_data)
    reloaded = PicklePatcher.loads(dumped)

    assert reloaded == original_data
    # Everything was pickleable, so no placeholders should appear.

def test_picklepatch_with_socket():
    """
    Test that a data structure containing a raw socket is replaced by
    PicklePlaceholder rather than raising an error.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_with_socket = {
        "safe_value": 123,
        "raw_socket": s,
    }

    dumped = PicklePatcher.dumps(data_with_socket)
    reloaded = PicklePatcher.loads(dumped)

    # We expect "raw_socket" to be replaced by a placeholder
    assert isinstance(reloaded, dict)
    assert reloaded["safe_value"] == 123
    assert isinstance(reloaded["raw_socket"], PicklePlaceholder)

    # Attempting to use or access attributes => AttributeError 
    # (not RuntimeError as in original tests, our implementation uses AttributeError)
    with pytest.raises(AttributeError):
        reloaded["raw_socket"].recv(1024)


def test_picklepatch_deeply_nested():
    """
    Test that deep nesting with unpicklable objects works correctly.
    """
    # Create a deeply nested structure with an unpicklable object
    deep_nested = {
        "level1": {
            "level2": {
                "level3": {
                    "normal": "value",
                    "socket": socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                }
            }
        }
    }

    dumped = PicklePatcher.dumps(deep_nested)
    reloaded = PicklePatcher.loads(dumped)

    # We should be able to access the normal value
    assert reloaded["level1"]["level2"]["level3"]["normal"] == "value"

    # The socket should be replaced with a placeholder
    assert isinstance(reloaded["level1"]["level2"]["level3"]["socket"], PicklePlaceholder)

def test_picklepatch_class_with_unpicklable_attr():
    """
    Test that a class with an unpicklable attribute works correctly.
    """
    class TestClass:
        def __init__(self):
            self.normal = "normal value"
            self.unpicklable = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    obj = TestClass()

    dumped = PicklePatcher.dumps(obj)
    reloaded = PicklePatcher.loads(dumped)

    # Normal attribute should be preserved
    assert reloaded.normal == "normal value"

    # Unpicklable attribute should be replaced with a placeholder
    assert isinstance(reloaded.unpicklable, PicklePlaceholder)




def test_picklepatch_with_database_connection():
    """
    Test that a data structure containing a database connection is replaced
    by PicklePlaceholder rather than raising an error.
    """
    # SQLite connection - not pickleable
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    data_with_db = {
        "description": "Database connection",
        "connection": conn,
        "cursor": cursor,
    }

    dumped = PicklePatcher.dumps(data_with_db)
    reloaded = PicklePatcher.loads(dumped)

    # Both connection and cursor should become placeholders
    assert isinstance(reloaded, dict)
    assert reloaded["description"] == "Database connection"
    assert isinstance(reloaded["connection"], PicklePlaceholder)
    assert isinstance(reloaded["cursor"], PicklePlaceholder)

    # Attempting to use attributes => AttributeError
    with pytest.raises(AttributeError):
        reloaded["connection"].execute("SELECT 1")


def test_picklepatch_with_generator():
    """
    Test that a data structure containing a generator is replaced by
    PicklePlaceholder rather than raising an error.
    """

    def simple_generator():
        yield 1
        yield 2
        yield 3

    # Create a generator
    gen = simple_generator()

    # Put it in a data structure
    data_with_generator = {
        "description": "Contains a generator",
        "generator": gen,
        "normal_list": [1, 2, 3]
    }

    dumped = PicklePatcher.dumps(data_with_generator)
    reloaded = PicklePatcher.loads(dumped)

    # Generator should be replaced with a placeholder
    assert isinstance(reloaded, dict)
    assert reloaded["description"] == "Contains a generator"
    assert reloaded["normal_list"] == [1, 2, 3]
    assert isinstance(reloaded["generator"], PicklePlaceholder)

    # Attempting to use the generator => AttributeError
    with pytest.raises(TypeError):
        next(reloaded["generator"])

    # Attempting to call methods on the generator => AttributeError
    with pytest.raises(AttributeError):
        reloaded["generator"].send(None)