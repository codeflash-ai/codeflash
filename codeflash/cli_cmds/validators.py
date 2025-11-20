from textual.validation import ValidationResult, Validator


class APIKeyValidator(Validator):
    """Validates Codeflash API key format (not authenticity - that's done async on submit)."""

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("API key cannot be empty")

        if not value.startswith("cf-"):
            return self.failure("API key must start with 'cf-' prefix")

        return self.success()
