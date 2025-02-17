

class CharacterRemover:
    def __init__(self):
        self.version = "0.1"
        self._translation_table = dict.fromkeys(
            range(32)
        )  # Create a translation table removing control characters 0x00-0x1F
        self._translation_table.update({127: None})  # Also remove 0x7F (DEL)

    def remove_control_characters(self, s) -> str:
        """Remove control characters from the string."""
        return s.translate(self._translation_table) if s else ""
