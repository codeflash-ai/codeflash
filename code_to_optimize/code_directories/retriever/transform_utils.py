from code_to_optimize.code_directories.retriever.utils import DataProcessor


class DataTransformer:
    def __init__(self):
        self.data = None

    def transform(self, data):
        self.data = data
        return self.data

    def transform_using_own_method(self, data):
        return self.transform(data)

    def transform_using_same_file_function(self, data):
        return update_data(data)

    def transform_data_all_same_file(self, data):
        new_data = update_data(data)
        return self.transform_using_own_method(new_data)

    def circular_dependency(self, data):
        return DataProcessor().circular_dependency(data)


def update_data(data):
    return data + " updated"
