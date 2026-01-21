"""Abstract base class for code normalizers.

Code normalizers transform source code into a canonical form for duplicate detection.
They normalize variable names, remove comments/docstrings, and produce consistent output
that can be compared across different implementations of the same algorithm.
"""
# TODO:{claude} move to base.py in language folder
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

    """Abstract base class for language-specific code normalizers.

    Subclasses must implement the normalize() method for their specific language.
    The normalization should:
    - Normalize local variable names to canonical forms (var_0, var_1, etc.)
    - Preserve function names, class names, parameters, and imports
    - Remove or normalize comments and docstrings
    - Produce consistent output for structurally identical code

    Example:
        >>> normalizer = PythonNormalizer()
        >>> code1 = "def foo(x): y = x + 1; return y"
        >>> code2 = "def foo(x): z = x + 1; return z"
        >>> normalizer.normalize(code1) == normalizer.normalize(code2)
        True
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language this normalizer handles."""
        ...

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return file extensions this normalizer can handle."""
        return ()

    @abstractmethod
    def normalize(self, code: str) -> str:
        """Normalize code to a canonical form for comparison.

        Args:
            code: Source code to normalize

        Returns:
            Normalized representation of the code
        """
        ...

    @abstractmethod
    def normalize_for_hash(self, code: str) -> str:
        """Normalize code optimized for hashing/fingerprinting.

        This may return a more compact representation than normalize().

        Args:
            code: Source code to normalize

        Returns:
            Normalized representation suitable for hashing
        """
        ...

    def are_duplicates(self, code1: str, code2: str) -> bool:
        """Check if two code segments are duplicates after normalization.

        Args:
            code1: First code segment
            code2: Second code segment

        Returns:
            True if codes are structurally identical
        """
        try:
            normalized1 = self.normalize_for_hash(code1)
            normalized2 = self.normalize_for_hash(code2)
            return normalized1 == normalized2
        except Exception:
            return False

    def get_fingerprint(self, code: str) -> str:
        """Generate a fingerprint hash for normalized code.

        Args:
            code: Source code to fingerprint

        Returns:
            SHA-256 hash of normalized code
        """
        import hashlib

        normalized = self.normalize_for_hash(code)
        return hashlib.sha256(normalized.encode()).hexdigest()
