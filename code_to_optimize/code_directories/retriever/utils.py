import math

from transform_utils import DataTransformer

GLOBAL_VAR = 10


class DataProcessor:
    """A class for processing data."""

    number = 1

    def __init__(self, default_prefix: str = "PREFIX_"):
        """Initialize the DataProcessor with a default prefix."""
        self.default_prefix = default_prefix
        self.number += math.log(self.number)

    def __repr__(self) -> str:
        """Return a string representation of the DataProcessor."""
        return f"DataProcessor(default_prefix={self.default_prefix!r})"

    def process_data(self, raw_data: str) -> str:
        """Process raw data by converting it to uppercase."""
        return raw_data.upper()

    def add_prefix(self, data: str, prefix: str = "PREFIX_") -> str:
        """Add a prefix to the processed data."""
        return prefix + data

    def do_something(self):
        print("something")

    def transform_data(self, data: str) -> str:
        """Transform the processed data"""
        return DataTransformer().transform(data)

    def transform_data_own_method(self, data: str) -> str:
        """Transform the processed data using own method"""
        return DataTransformer().transform_using_own_method(data)

    def transform_data_same_file_function(self, data: str) -> str:
        """Transform the processed data using a function from the same file"""
        return DataTransformer().transform_using_same_file_function(data)

    def circular_dependency(self, data: str) -> str:
        """Test circular dependency"""
        return DataTransformer().circular_dependency(data)
