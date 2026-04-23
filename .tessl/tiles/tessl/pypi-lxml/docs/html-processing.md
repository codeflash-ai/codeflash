# HTML Processing

Specialized HTML document processing with web-specific features including lenient parsing, form handling, link processing, CSS class manipulation, and HTML5 support. The html module provides a high-level interface optimized for working with HTML documents in web applications.

## Capabilities

### HTML Document Parsing

Parse HTML documents with lenient parsing that handles malformed HTML gracefully.

```python { .api }
def parse(filename_or_url, parser=None, base_url=None, **kwargs):
    """
    Parse HTML document from file or URL.
    
    Args:
        filename_or_url: Path to file or URL to parse
        parser: HTMLParser instance (optional)
        base_url: Base URL for resolving relative links (optional)
        **kwargs: Additional arguments passed to parser
    
    Returns:
        ElementTree: Parsed HTML document tree
    """

def document_fromstring(html, parser=None, ensure_head_body=False, base_url=None):
    """
    Parse complete HTML document from string.
    
    Args:
        html: str or bytes containing HTML content
        parser: HTMLParser instance (optional)
        ensure_head_body: Ensure document has <head> and <body> elements
        base_url: Base URL for resolving relative references
    
    Returns:
        Element: Root <html> element
    """

def fragment_fromstring(html, create_parent=False, tag=None, base_url=None, parser=None):
    """
    Parse HTML fragment from string.
    
    Args:
        html: str or bytes containing HTML fragment
        create_parent: Wrap fragment in parent element
        tag: Parent tag name if create_parent=True
        base_url: Base URL for resolving relative references
        parser: HTMLParser instance (optional)
    
    Returns:
        Element: Fragment root element or parent element
    """

def fragments_fromstring(html, no_leading_text=False, base_url=None, parser=None):
    """
    Parse HTML string into list of elements and text.
    
    Args:
        html: str or bytes containing HTML fragments
        no_leading_text: Exclude leading text before first element
        base_url: Base URL for resolving relative references
        parser: HTMLParser instance (optional)
    
    Returns:
        list: Elements and text strings from parsed content
    """

def fromstring(html, base_url=None, parser=None):
    """
    Intelligently parse HTML as document or fragment.
    
    Args:
        html: str or bytes containing HTML content
        base_url: Base URL for resolving relative references
        parser: HTMLParser instance (optional)
    
    Returns:
        Element: Root element (document or fragment)
    """
```

### HTML Element Classes

HTML-specific element classes with web functionality.

```python { .api }
class HtmlElement:
    """Base HTML element class with HTML-specific methods."""
    
    # CSS class manipulation
    def get_class(self):
        """Get CSS classes as set-like object."""
    
    def set_class(self, classes):
        """Set CSS classes from string or iterable."""
    
    classes = property(get_class, set_class)
    
    # Link processing
    def make_links_absolute(self, base_url=None, resolve_base_href=True):
        """Make all relative links absolute."""
    
    def resolve_base_href(self, handle_failures=True):
        """Apply base href to relative links."""
    
    def iterlinks(self):
        """Iterate over all links in element."""
    
    def rewrite_links(self, link_repl_func, resolve_base_href=True, base_href=None):
        """Rewrite links using callback function."""
    
    # Content extraction
    def text_content(self):
        """Get all text content with whitespace normalized."""
    
    def drop_tree(self):
        """Remove element and children from document."""
    
    def drop_tag(self):
        """Remove element tag but keep children."""
    
    # Form-related methods (for form elements)
    @property
    def forms(self):
        """List of form elements in document."""
    
    @property
    def body(self):
        """Document body element (for document root)."""

class HtmlComment(HtmlElement):
    """HTML comment element."""

class HtmlEntity(HtmlElement):  
    """HTML entity element."""

class HtmlProcessingInstruction(HtmlElement):
    """HTML processing instruction element."""
```

### Form Handling

Specialized classes for working with HTML forms and form elements.

```python { .api }
class FormElement(HtmlElement):
    """HTML form element with submission capabilities."""
    
    @property
    def inputs(self):
        """Dictionary-like access to form inputs."""
    
    @property  
    def fields(self):
        """Dictionary of form field names to elements."""
    
    @property
    def action(self):
        """Form action URL."""
    
    @property
    def method(self):
        """Form submission method (GET/POST)."""
    
    def form_values(self):
        """Get list of (name, value) pairs for form submission."""
    
    def _name_values(self):
        """Internal method for getting form data."""

class InputElement(HtmlElement):
    """HTML input element."""
    
    @property
    def name(self):
        """Input name attribute."""
    
    @property
    def value(self):
        """Input value."""
    
    @value.setter
    def value(self, value):
        """Set input value."""
    
    @property
    def type(self):
        """Input type (text, password, checkbox, etc.)."""
    
    @property
    def checked(self):
        """Checked state for checkbox/radio inputs."""
    
    @checked.setter
    def checked(self, checked):
        """Set checked state."""

class SelectElement(HtmlElement):
    """HTML select element."""
    
    @property
    def value(self):
        """Selected value(s)."""
    
    @value.setter  
    def value(self, value):
        """Set selected value(s)."""
    
    @property
    def value_options(self):
        """List of possible values."""
    
    @property
    def multiple(self):
        """Multiple selection enabled."""

class TextareaElement(HtmlElement):
    """HTML textarea element."""
    
    @property
    def value(self):
        """Textarea content."""
    
    @value.setter
    def value(self, value):
        """Set textarea content."""

class LabelElement(HtmlElement):
    """HTML label element."""
    
    @property
    def for_element(self):
        """Associated form element."""
```

### Link Processing

Functions for processing and manipulating links in HTML documents.

```python { .api }
def make_links_absolute(element, base_url=None, resolve_base_href=True, handle_failures=True):
    """
    Convert relative links to absolute URLs.
    
    Args:
        element: HTML element or document
        base_url: Base URL for resolving relative links
        resolve_base_href: Process <base href> elements first
        handle_failures: Continue on URL resolution errors
    """

def resolve_base_href(element, handle_failures=True):
    """
    Apply <base href> elements to relative links.
    
    Args:
        element: HTML element or document
        handle_failures: Continue on URL resolution errors
    """

def iterlinks(element):
    """
    Iterate over all links in HTML element.
    
    Args:
        element: HTML element or document
    
    Yields:
        tuple: (element, attribute, link, pos) for each link
    """

def rewrite_links(element, link_repl_func, resolve_base_href=True, base_href=None):
    """
    Rewrite links using callback function.
    
    Args:
        element: HTML element or document
        link_repl_func: Function to transform URLs
        resolve_base_href: Process <base href> elements first
        base_href: Override base URL
    """

def find_rel_links(element, rel):
    """
    Find links with specified rel attribute.
    
    Args:
        element: HTML element or document
        rel: rel attribute value to match
    
    Returns:
        list: Elements with matching rel attribute
    """

def find_class(element, class_name):
    """
    Find elements with specified CSS class.
    
    Args:
        element: HTML element or document
        class_name: CSS class name to match
    
    Returns:
        list: Elements with matching class
    """
```

### CSS Class Management

Utility classes for managing CSS classes on HTML elements.

```python { .api }
class Classes:
    """Set-like interface for CSS classes."""
    
    def __init__(self, element):
        """Create class manager for element."""
    
    def add(self, *classes):
        """Add CSS classes."""
    
    def discard(self, class_name):
        """Remove CSS class if present."""
    
    def remove(self, class_name):
        """Remove CSS class (raises KeyError if not present)."""
    
    def update(self, classes):
        """Add multiple classes from iterable."""
    
    def clear(self):
        """Remove all classes."""
    
    def __contains__(self, class_name):
        """Test if class is present."""
    
    def __iter__(self):
        """Iterate over classes."""
    
    def __len__(self):
        """Number of classes."""
```

### HTML Serialization

Convert HTML elements and documents to strings with HTML-specific formatting.

```python { .api }
def tostring(doc, pretty_print=False, include_meta_content_type=False, 
             encoding=None, method="html", with_tail=True, doctype=None):
    """
    Serialize HTML element or document to string.
    
    Args:
        doc: HTML element or document
        pretty_print: Format output with whitespace
        include_meta_content_type: Add meta charset tag
        encoding: Output encoding ('unicode' for str)
        method: Serialization method (usually 'html')
        with_tail: Include tail text
        doctype: Document type declaration
    
    Returns:
        str or bytes: Serialized HTML
    """
```

### Form Submission

Submit HTML forms programmatically.

```python { .api }
def submit_form(form, extra_values=None, open_http=None):
    """
    Submit HTML form and return response.
    
    Args:
        form: FormElement to submit
        extra_values: Additional form values as dict
        open_http: Function to handle HTTP request
    
    Returns:
        Response from form submission
    """
```

### Utility Functions

Additional HTML processing utilities.

```python { .api }
def Element(tag, attrib=None, nsmap=None, **extra):
    """
    Create HTML element.
    
    Args:
        tag: Element tag name
        attrib: Attribute dictionary
        nsmap: Namespace mapping (rarely used for HTML)
        **extra: Additional attributes
    
    Returns:  
        HtmlElement: New HTML element
    """

def open_in_browser(doc, encoding=None):
    """
    Open HTML document in web browser.
    
    Args:
        doc: HTML element or document
        encoding: Character encoding for temporary file
    """
```

### Sub-modules

Additional HTML processing functionality in sub-modules.

```python { .api }
# HTML definitions and constants
import lxml.html.defs

# HTML element builder
import lxml.html.builder

# HTML document comparison and diffing  
import lxml.html.diff

# Form filling utilities
import lxml.html.formfill

# HTML cleaning and sanitization
import lxml.html.clean

# BeautifulSoup compatibility
import lxml.html.soupparser

# HTML5 parsing (requires html5lib)
import lxml.html.html5parser
```

## Usage Examples

### Basic HTML Processing

```python
from lxml import html

# Parse HTML document
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sample Page</title>
    <base href="https://example.com/">
</head>
<body>
    <div class="header">
        <h1>Welcome</h1>
        <nav>
            <a href="/home">Home</a>
            <a href="/about">About</a>
            <a href="contact.html">Contact</a>
        </nav>
    </div>
    <div class="content main-content">
        <p>This is the main content.</p>
        <img src="images/logo.png" alt="Logo">
    </div>
</body>
</html>
'''

doc = html.fromstring(html_content)

# Find elements by CSS class
header = html.find_class(doc, 'header')[0]
content_divs = html.find_class(doc, 'content')

# Work with CSS classes
content_div = content_divs[0]
print(content_div.classes)  # {'content', 'main-content'}
content_div.classes.add('highlighted')
content_div.classes.discard('main-content')

# Process links
html.make_links_absolute(doc, base_url='https://mysite.com')
for element, attribute, link, pos in html.iterlinks(doc):
    print(f"{element.tag}.{attribute}: {link}")

# Get text content
title = doc.find('.//title').text_content()
print(f"Page title: {title}")
```

### Form Processing

```python
from lxml import html

# HTML with form
form_html = '''
<html>
<body>
    <form action="/login" method="post">
        <input type="text" name="username" value="john">
        <input type="password" name="password" value="">
        <input type="checkbox" name="remember" checked>
        <select name="role">
            <option value="user">User</option>
            <option value="admin" selected>Admin</option>
        </select>
        <textarea name="comments">Default text</textarea>
        <button type="submit">Login</button>
    </form>
</body>
</html>
'''

doc = html.fromstring(form_html)
form = doc.forms[0]

# Access form properties
print(f"Action: {form.action}")
print(f"Method: {form.method}")

# Work with form fields
print("Form fields:")
for name, element in form.fields.items():
    if hasattr(element, 'value'):
        print(f"  {name}: {element.value}")
    elif hasattr(element, 'checked'):
        print(f"  {name}: {'checked' if element.checked else 'unchecked'}")

# Modify form values
form.fields['username'].value = 'alice'
form.fields['password'].value = 'secret123'
form.fields['remember'].checked = False
form.fields['role'].value = 'user'

# Get form data for submission
form_data = form.form_values()
print("Form data:", dict(form_data))
```

### Link Manipulation

```python
from lxml import html

html_content = '''
<div>
    <a href="/internal">Internal Link</a>
    <a href="http://external.com">External Link</a>
    <img src="images/photo.jpg" alt="Photo">
    <link rel="stylesheet" href="styles/main.css">
</div>
'''

doc = html.fragment_fromstring(html_content)

# Make links absolute
html.make_links_absolute(doc, base_url='https://mysite.com')

# Rewrite specific links
def rewrite_image_links(url):
    if url.endswith(('.jpg', '.png', '.gif')):
        return f"https://cdn.mysite.com/{url.lstrip('/')}"
    return url

html.rewrite_links(doc, rewrite_image_links)

# Find specific link types
stylesheets = html.find_rel_links(doc, 'stylesheet')
for link in stylesheets:
    print(f"Stylesheet: {link.get('href')}")

print(html.tostring(doc, encoding='unicode'))
```

### Content Extraction and Modification

```python
from lxml import html

html_content = '''
<article>
    <h1>Article Title</h1>
    <div class="meta">
        <span class="author">John Doe</span>
        <span class="date">2023-12-07</span>
    </div>
    <div class="content">
        <p>First paragraph with <a href="link1.html">a link</a>.</p>
        <p>Second paragraph with <strong>bold text</strong>.</p>
        <div class="sidebar">Sidebar content</div>
    </div>
</article>
'''

doc = html.fromstring(html_content)

# Extract text content
title = doc.find('.//h1').text_content()
author = html.find_class(doc, 'author')[0].text_content()
content_text = html.find_class(doc, 'content')[0].text_content()

print(f"Title: {title}")
print(f"Author: {author}")
print(f"Content: {content_text[:100]}...")

# Remove unwanted elements
sidebar = html.find_class(doc, 'sidebar')[0]
sidebar.drop_tree()  # Remove element and children

# Remove tags but keep content
for strong in doc.xpath('.//strong'):
    strong.drop_tag()  # Remove <strong> tags but keep text

print(html.tostring(doc, pretty_print=True, encoding='unicode'))
```

### CSS Class Management

```python
from lxml import html

html_content = '<div class="content main highlighted"></div>'
element = html.fragment_fromstring(html_content)

# Work with classes as a set
classes = element.classes
print(f"Initial classes: {set(classes)}")

# Add and remove classes
classes.add('active')
classes.discard('highlighted')
classes.update(['responsive', 'mobile-friendly'])

print(f"Final classes: {set(classes)}")
print(f"Has 'active': {'active' in classes}")
print(f"Number of classes: {len(classes)}")

# Convert back to HTML
print(html.tostring(element, encoding='unicode'))
```