from textual.validation import ValidationResult, Validator


class APIKeyValidator(Validator):
    """Validates Codeflash API key format."""

    def validate(self, value: str) -> ValidationResult:
        """Check if API key is valid (starts with 'cf-')."""
        if not value:
            return self.failure("API key cannot be empty")

        if not value.startswith("cf-"):
            return self.failure("API key must start with 'cf-' prefix")

        return self.success()
