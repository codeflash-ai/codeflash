# Internationalization

Activate and manage localization for 25+ supported languages with locale-specific number formatting, date formatting, and text translations throughout the humanize library.

## Capabilities

### Locale Activation

Activate internationalization for a specific locale, loading the appropriate translation files and configuring locale-specific formatting.

```python { .api }
def activate(
    locale: str | None, 
    path: str | os.PathLike[str] | None = None
) -> gettext.NullTranslations:
    """
    Activate internationalization for a specific locale.
    
    Args:
        locale: Language code (e.g., 'en_GB', 'fr_FR', 'de_DE') or None for default
        path: Custom path to search for locale files (optional)
    
    Returns:
        Translation object for the activated locale
    
    Raises:
        Exception: If locale folder cannot be found and path not provided
    
    Examples:
        >>> activate('fr_FR')  # Activate French
        >>> activate('de_DE')  # Activate German
        >>> activate(None)     # Deactivate (same as deactivate())
    """
```

### Locale Deactivation

Deactivate internationalization, returning to the default English behavior.

```python { .api }
def deactivate() -> None:
    """
    Deactivate internationalization, returning to default English.
    
    Example:
        >>> activate('fr_FR')
        >>> intword(1000000)  # Returns French text
        >>> deactivate()
        >>> intword(1000000)  # Returns '1.0 million'
    """
```

### Number Formatting Separators

Get locale-specific thousands and decimal separators for number formatting.

```python { .api }
def thousands_separator() -> str:
    """
    Return the thousands separator for current locale.
    
    Returns:
        Thousands separator character (default: ',')
    
    Examples:
        >>> activate('en_US')
        >>> thousands_separator()
        ','
        >>> activate('fr_FR')
        >>> thousands_separator()
        ' '
        >>> activate('de_DE')
        >>> thousands_separator()
        '.'
    """

def decimal_separator() -> str:
    """
    Return the decimal separator for current locale.
    
    Returns:
        Decimal separator character (default: '.')
    
    Examples:
        >>> activate('en_US')
        >>> decimal_separator()
        '.'
        >>> activate('de_DE')
        >>> decimal_separator()
        ','
        >>> activate('fr_FR')
        >>> decimal_separator()
        '.'
    """
```

## Supported Languages

The humanize library includes translations for over 25 languages:

- **Arabic** (ar)
- **Basque** (eu)
- **Bengali** (bn)
- **Brazilian Portuguese** (pt_BR)
- **Catalan** (ca)
- **Danish** (da)
- **Dutch** (nl)
- **Esperanto** (eo)
- **European Portuguese** (pt)
- **Finnish** (fi)
- **French** (fr_FR)
- **German** (de_DE)
- **Greek** (el)
- **Hebrew** (he)
- **Indonesian** (id)
- **Italian** (it_IT)
- **Japanese** (ja)
- **Klingon** (tlh)
- **Korean** (ko)
- **Norwegian** (no)
- **Persian** (fa)
- **Polish** (pl)
- **Russian** (ru)
- **Simplified Chinese** (zh_CN)
- **Slovak** (sk)
- **Slovenian** (sl)
- **Spanish** (es)
- **Swedish** (sv)
- **Turkish** (tr)
- **Ukrainian** (uk)
- **Vietnamese** (vi)

## Usage Examples

### Basic Localization

```python
import humanize

# Default English behavior
print(humanize.intword(1000000))  # "1.0 million"
print(humanize.naturaltime(3600, future=True))  # "an hour from now"

# Activate French
humanize.activate('fr_FR')
print(humanize.intword(1000000))  # "1.0 million" (French translation)
print(humanize.naturaltime(3600, future=True))  # French equivalent

# Return to English
humanize.deactivate()
print(humanize.intword(1000000))  # "1.0 million"
```

### Number Formatting with Locales

```python
import humanize

# German locale - uses period for thousands, comma for decimal
humanize.activate('de_DE')
print(humanize.intcomma(1234567.89))  # Uses German separators
print(humanize.thousands_separator())  # '.'
print(humanize.decimal_separator())   # ','

# French locale - uses space for thousands
humanize.activate('fr_FR')  
print(humanize.intcomma(1234567.89))  # Uses French separators
print(humanize.thousands_separator())  # ' '
print(humanize.decimal_separator())   # '.'
```

### Context Manager Pattern

For temporary locale changes, you can create a simple context manager:

```python
import humanize
from contextlib import contextmanager

@contextmanager
def temporary_locale(locale):
    """Temporarily activate a locale."""
    original = humanize.get_translation()  # Would need to be implemented
    try:
        humanize.activate(locale)
        yield
    finally:
        humanize.deactivate()

# Usage
with temporary_locale('de_DE'):
    print(humanize.intcomma(1234567))  # German formatting
print(humanize.intcomma(1234567))      # Back to default
```

### Ordinal Gender Support

Some languages support gendered ordinals:

```python
import humanize

# English (no gender distinction)
print(humanize.ordinal(1))  # "1st"

# Languages with gender support (theoretical example)
humanize.activate('some_locale')
print(humanize.ordinal(1, gender='male'))    # Masculine form
print(humanize.ordinal(1, gender='female'))  # Feminine form
```

## Locale File Structure

Locale files are stored in the package's locale directory using the standard gettext format:

```
locale/
├── fr_FR/
│   └── LC_MESSAGES/
│       ├── humanize.po
│       └── humanize.mo
├── de_DE/
│   └── LC_MESSAGES/
│       ├── humanize.po
│       └── humanize.mo
└── ...
```

## Custom Locale Paths

You can specify custom paths for locale files:

```python
import humanize

# Use custom locale directory
humanize.activate('custom_locale', path='/path/to/custom/locales')
```

## Implementation Details

```python { .api }
# Internal translation functions (not in __all__ but accessible)
def get_translation() -> gettext.NullTranslations: ...
def _gettext(message: str) -> str: ...
def _pgettext(msgctxt: str, message: str) -> str: ...
def _ngettext(message: str, plural: str, num: int) -> str: ...
def _gettext_noop(message: str) -> str: ...
def _ngettext_noop(singular: str, plural: str) -> tuple[str, str]: ...
```

## Error Handling

- If a locale cannot be found, an exception is raised unless a custom path is provided
- If locale files are corrupted or missing, the function falls back to English
- Invalid locale codes are handled gracefully
- Thread-local storage ensures locale settings don't interfere between threads

## Thread Safety

The internationalization system uses thread-local storage, making it safe to use different locales in different threads simultaneously without interference.