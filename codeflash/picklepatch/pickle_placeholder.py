class PicklePlaceholder:
    """A placeholder for an object that couldn't be pickled.

    When unpickled, any attempt to access attributes or call methods on this
    placeholder will raise an informative exception.
    """

    def __init__(self, obj_type, obj_str, error_msg, path=None):
        """Initialize a placeholder for an unpicklable object.

        Args:
            obj_type (str): The type name of the original object
            obj_str (str): String representation of the original object
            error_msg (str): The error message that occurred during pickling
            path (list, optional): Path to this object in the original object graph

        """
        # Store these directly in __dict__ to avoid __getattr__ recursion
        self.__dict__["obj_type"] = obj_type
        self.__dict__["obj_str"] = obj_str
        self.__dict__["error_msg"] = error_msg
        self.__dict__["path"] = path if path is not None else []

    def __getattr__(self, name):
        """Raise an error when any attribute is accessed."""
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        raise AttributeError(
            f"Cannot access attribute '{name}' on unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )

    def __setattr__(self, name, value):
        """Prevent setting attributes."""
        self.__getattr__(name)  # This will raise an AttributeError

    def __call__(self, *args, **kwargs):
        """Raise an error when the object is called."""
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        raise TypeError(
            f"Cannot call unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )

    def __repr__(self):
        """Return a string representation of the placeholder."""
        try:
            path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root"
            return f"<PicklePlaceholder at {path_str}: {self.__dict__['obj_type']} {self.__dict__['obj_str']}>"
        except:
            return "<PicklePlaceholder: (error displaying details)>"

    def __str__(self):
        """Return a string representation of the placeholder."""
        return self.__repr__()

    def __reduce__(self):
        """Make sure pickling of the placeholder itself works correctly."""
        return (
            PicklePlaceholder,
            (
                self.__dict__["obj_type"],
                self.__dict__["obj_str"],
                self.__dict__["error_msg"],
                self.__dict__["path"]
            )
        )
