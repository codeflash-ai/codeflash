import os

class UnformattedExampleClass(object):
    def __init__(
        self,
        name,
        age=    None,
        email=  None,
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
        self.   address = address
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
        data_list = [   item for item in data_list if filter_func(item)]
    if transform_func:
        data_list = [transform_func(item) for item in data_list]
    if sort_key:
        data_list = sorted(data_list, key=sort_key, reverse=reverse)
    return data_list

