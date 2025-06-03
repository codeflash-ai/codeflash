class CharacterRemover:
    def __init__(self):
        self.version = "0.1"
        # Build translation table once in init.
        self._ctrl_table = self._make_ctrl_table()

    def remove_control_characters(self, s) -> str:
        """Remove control characters from the string."""
        return s.translate(self._ctrl_table) if s else ""

    def _make_ctrl_table(self):
        # Map delete (ASCII 127) and 0-31 to None
        ctrl_chars = dict.fromkeys(range(32), None)
        ctrl_chars[127] = None
        return str.maketrans(ctrl_chars)
