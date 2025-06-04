import os, sys, json, datetime, math, random
import requests
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

# This is a poorly formatted Python file with many style violations


class UnformattedExampleClass(object):
    def __init__(
        self,
        name,
        age=None,
        email=None,
        phone=None,
        address=None,
        city=None,
        state=None,
        zip_code=None,
    ):
        self.name = name
        self.age = age
        self.email = email
        self.phone = phone
        self.address = address
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.data = {"name": name, "age": age, "email": email}

    def get_info(self):
        return f"Name: {self.name}, Age: {self.age}"

    def update_data(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.data.update(kwargs)


def process_data(
    data_list, filter_func=None, transform_func=None, sort_key=None, reverse=False
):
    if not data_list:
        return []
    if filter_func:
        data_list = [item for item in data_list if filter_func(item)]
    if transform_func:
        data_list = [transform_func(item) for item in data_list]
    if sort_key:
        data_list = sorted(data_list, key=sort_key, reverse=reverse)
    return data_list


def calculate_statistics(numbers):
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    median = sorted(numbers)[len(numbers) // 2]
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = math.sqrt(variance)
    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "std_dev": std_dev,
        "min": min(numbers),
        "max": max(numbers),
    }
