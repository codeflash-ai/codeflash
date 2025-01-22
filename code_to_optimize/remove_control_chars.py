import re


class CharacterRemover:
    def __init__(self):
        self.version = "0.1"

    def remove_control_characters(self, s) -> str:
        """Remove control characters from the string."""
        return re.sub("[\\x00-\\x1F\\x7F]", "", s) if s else ""
