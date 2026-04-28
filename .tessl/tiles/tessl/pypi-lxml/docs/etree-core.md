# Core XML/HTML Processing

Comprehensive ElementTree-compatible API for XML and HTML document parsing, manipulation, and serialization. This module provides the foundation for all lxml functionality with full standards compliance, namespace support, and high-performance processing.

## Capabilities

### Document Parsing

Parse XML and HTML documents from strings, files, URLs, or file-like objects with configurable parsers and error handling.

```python { .api }
def parse(source, parser=None, base_url=None):
    """
    Parse XML/HTML document from file, URL, or file-like object.
    
    Args:
        source: File path, URL, file-like object, or filename
        parser: XMLParser or HTMLParser instance (optional)
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        ElementTree: Parsed document tree
    """

def fromstring(text, parser=None, base_url=None):
    """
    Parse XML/HTML document from string.
    
    Args:
        text: str or bytes containing XML/HTML content
        parser: XMLParser or HTMLParser instance (optional)
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        Element: Root element of parsed document
    """

def XML(text, parser=None, base_url=None):
    """
    Parse XML string with validation enabled by default.
    
    Args:
        text: str or bytes containing XML content
        parser: XMLParser instance (optional)
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        Element: Root element of parsed XML
    """

def HTML(text, parser=None, base_url=None):
    """
    Parse HTML string with lenient parsing.
    
    Args:
        text: str or bytes containing HTML content
        parser: HTMLParser instance (optional)
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        Element: Root element of parsed HTML
    """
```

### Incremental Parsing

Memory-efficient parsing for large documents using event-driven processing.

```python { .api }
def iterparse(source, events=None, tag=None, attribute_defaults=False, 
              dtd_validation=False, load_dtd=False, no_network=True, 
              remove_blank_text=False, remove_comments=False, 
              remove_pis=False, encoding=None, huge_tree=False, 
              schema=None):
    """
    Incrementally parse XML document yielding (event, element) pairs.
    
    Args:
        source: File path, URL, or file-like object
        events: tuple of events to report ('start', 'end', 'start-ns', 'end-ns')
        tag: str or sequence of tag names to filter
        
    Yields:
        tuple: (event, element) pairs during parsing
    """

def iterwalk(element_or_tree, events=('end',), tag=None):
    """
    Walk through existing element tree yielding events.
    
    Args:
        element_or_tree: Element or ElementTree to walk
        events: tuple of events to report ('start', 'end')
        tag: str or sequence of tag names to filter
        
    Yields:
        tuple: (event, element) pairs during traversal
    """
```

### Element Creation and Manipulation

Create and modify XML/HTML elements with full attribute and content support.

```python { .api }
class Element:
    """XML/HTML element with tag, attributes, text, and children."""
    
    def __init__(self, tag, attrib=None, nsmap=None, **extra):
        """
        Create new element.
        
        Args:
            tag: Element tag name (str or QName)
            attrib: dict of attributes (optional)
            nsmap: dict mapping namespace prefixes to URIs (optional)
            **extra: Additional attributes as keyword arguments
        """
    
    # Element properties
    tag: str                    # Element tag name
    text: str | None           # Text content before first child
    tail: str | None           # Text content after element
    attrib: dict[str, str]     # Element attributes
    nsmap: dict[str, str]      # Namespace mapping
    sourceline: int | None     # Source line number (if available)
    
    # Tree navigation
    def find(self, path, namespaces=None):
        """Find first child element matching path."""
    
    def findall(self, path, namespaces=None):
        """Find all child elements matching path."""
    
    def iterfind(self, path, namespaces=None):
        """Iterate over child elements matching path."""
    
    def findtext(self, path, default=None, namespaces=None):
        """Find text content of first matching child element."""
    
    def xpath(self, _path, namespaces=None, extensions=None, 
              smart_strings=True, **_variables):
        """Evaluate XPath expression on element."""
    
    # Tree modification
    def append(self, element):
        """Add element as last child."""
    
    def insert(self, index, element):
        """Insert element at specified position."""
    
    def remove(self, element):
        """Remove child element."""
    
    def clear(self):
        """Remove all children and attributes."""
    
    # Attribute access
    def get(self, key, default=None):
        """Get attribute value."""
    
    def set(self, key, value):
        """Set attribute value."""
    
    def keys(self):
        """Get attribute names."""
    
    def values(self):
        """Get attribute values."""
    
    def items(self):
        """Get (name, value) pairs for attributes."""

def SubElement(parent, tag, attrib=None, nsmap=None, **extra):
    """
    Create child element and add to parent.
    
    Args:
        parent: Parent Element
        tag: Child element tag name
        attrib: dict of attributes (optional)
        nsmap: dict of namespace mappings (optional)
        **extra: Additional attributes
    
    Returns:
        Element: New child element
    """
```

### Document Trees

Manage complete XML/HTML documents with document-level operations.

```python { .api }
class ElementTree:
    """Document tree containing root element and document info."""
    
    def __init__(self, element=None, file=None, parser=None):
        """
        Create document tree.
        
        Args:
            element: Root element (optional)
            file: File to parse (optional)
            parser: Parser instance (optional)
        """
    
    def getroot(self):
        """Get root element."""
    
    def setroot(self, root):
        """Set root element."""
    
    def parse(self, source, parser=None, base_url=None):
        """Parse document from source."""
    
    def write(self, file, encoding=None, xml_declaration=None, 
              default_namespace=None, method="xml", pretty_print=False,
              with_tail=True, standalone=None, compression=0, 
              exclusive=False, inclusive_ns_prefixes=None, 
              with_comments=True, strip_cdata=True):
        """Write document to file."""
    
    def xpath(self, _path, namespaces=None, extensions=None, 
              smart_strings=True, **_variables):
        """Evaluate XPath expression on document."""
    
    def xslt(self, _xslt, extensions=None, access_control=None, **_kw):
        """Apply XSLT transformation."""
    
    def relaxng(self, relaxng):
        """Validate against RelaxNG schema."""
    
    def xmlschema(self, xmlschema):
        """Validate against XML Schema."""
    
    def xinclude(self):
        """Process XInclude directives."""

    @property 
    def docinfo(self):
        """Document information (encoding, version, etc.)."""
```

### Serialization

Convert elements and trees to strings or bytes with formatting options.

```python { .api }
def tostring(element_or_tree, encoding=None, method="xml", 
             xml_declaration=None, pretty_print=False, with_tail=True, 
             standalone=None, doctype=None, exclusive=False, 
             inclusive_ns_prefixes=None, with_comments=True, 
             strip_cdata=True):
    """
    Serialize element or tree to string/bytes.
    
    Args:
        element_or_tree: Element or ElementTree to serialize
        encoding: Output encoding ('unicode' for str, bytes encoding for bytes)
        method: Serialization method ('xml', 'html', 'text', 'c14n')
        xml_declaration: Include XML declaration (bool or None for auto)
        pretty_print: Format output with whitespace (bool)
        with_tail: Include tail text (bool)
        doctype: Document type declaration (str)
        
    Returns:
        str or bytes: Serialized document
    """

def tostringlist(element_or_tree, encoding=None, method="xml", 
                 xml_declaration=None, pretty_print=False, with_tail=True, 
                 standalone=None, doctype=None, exclusive=False, 
                 inclusive_ns_prefixes=None, with_comments=True, 
                 strip_cdata=True):
    """Serialize to list of strings/bytes."""

def tounicode(element_or_tree, method="xml", pretty_print=False, 
              with_tail=True, doctype=None):
    """Serialize to unicode string."""

def dump(elem):
    """Debug dump element structure to stdout."""
```

### Parser Configuration

Configurable parsers for different XML/HTML processing needs.

```python { .api }
class XMLParser:
    """Configurable XML parser with validation and processing options."""
    
    def __init__(self, encoding=None, attribute_defaults=False,
                 dtd_validation=False, load_dtd=False, no_network=True,
                 ns_clean=False, recover=False, schema=None,
                 huge_tree=False, remove_blank_text=False,
                 resolve_entities=True, remove_comments=False,
                 remove_pis=False, strip_cdata=True, collect_ids=True,
                 target=None, compact=True):
        """
        Create XML parser with specified options.
        
        Args:
            encoding: Character encoding override
            attribute_defaults: Load default attributes from DTD
            dtd_validation: Enable DTD validation
            load_dtd: Load and parse DTD
            no_network: Disable network access
            recover: Enable error recovery
            huge_tree: Support very large documents
            remove_blank_text: Remove whitespace-only text nodes
            remove_comments: Remove comment nodes
            remove_pis: Remove processing instruction nodes
        """

class HTMLParser:
    """Lenient HTML parser with automatic error recovery."""
    
    def __init__(self, encoding=None, remove_blank_text=False,
                 remove_comments=False, remove_pis=False, 
                 strip_cdata=True, no_network=True, target=None,
                 schema=None, recover=True, compact=True):
        """Create HTML parser with specified options."""

def get_default_parser():
    """Get current default parser."""

def set_default_parser(parser):
    """Set global default parser."""
```

### Tree Manipulation Utilities

High-level functions for common tree modification operations.

```python { .api }
def cleanup_namespaces(tree_or_element):
    """Remove unused namespace declarations."""

def strip_attributes(tree_or_element, *attribute_names):
    """Remove specified attributes from all elements."""

def strip_elements(tree_or_element, *tag_names, with_tail=True):
    """Remove elements with specified tag names."""

def strip_tags(tree_or_element, *tag_names):
    """Remove tags but keep text content."""

def register_namespace(prefix, uri):
    """Register namespace prefix for serialization."""
```

### Node Type Classes

Specialized classes for different XML node types.

```python { .api }
class Comment:
    """XML comment node."""
    def __init__(self, text=None): ...

class ProcessingInstruction:
    """XML processing instruction node."""
    def __init__(self, target, text=None): ...
    
    @property
    def target(self) -> str: ...

class Entity:
    """XML entity reference node."""
    def __init__(self, name): ...
    
    @property
    def name(self) -> str: ...

class CDATA:
    """XML CDATA section."""
    def __init__(self, data): ...

# Factory functions
def Comment(text=None):
    """Create comment node."""

def ProcessingInstruction(target, text=None):
    """Create processing instruction node."""
    
PI = ProcessingInstruction  # Alias
```

## Usage Examples

### Basic XML Processing

```python
from lxml import etree

# Parse XML document
xml_data = '''<?xml version="1.0"?>
<catalog>
    <book id="1" category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <price currency="USD">12.99</price>
    </book>
    <book id="2" category="science">
        <title>A Brief History of Time</title>
        <author>Stephen Hawking</author>
        <year>1988</year>
        <price currency="USD">15.99</price>
    </book>
</catalog>'''

root = etree.fromstring(xml_data)

# Navigate and query
books = root.findall('book')
fiction_books = root.xpath('//book[@category="fiction"]')
titles = root.xpath('//title/text()')

# Modify content
new_book = etree.SubElement(root, 'book', id="3", category="mystery")
etree.SubElement(new_book, 'title').text = "The Murder Mystery"
etree.SubElement(new_book, 'author').text = "Agatha Christie"
etree.SubElement(new_book, 'year').text = "1934"
price_elem = etree.SubElement(new_book, 'price', currency="USD")
price_elem.text = "11.99"

# Serialize with formatting
output = etree.tostring(root, pretty_print=True, encoding='unicode')
print(output)
```

### HTML Document Processing

```python
from lxml import etree

# Parse HTML with XML parser (requires well-formed HTML)
html_data = '''<!DOCTYPE html>
<html>
<head>
    <title>Sample Page</title>
    <meta charset="UTF-8"/>
</head>
<body>
    <h1>Welcome</h1>
    <div class="content">
        <p>This is a paragraph.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
</body>
</html>'''

# Use HTML parser for lenient parsing
parser = etree.HTMLParser()
doc = etree.fromstring(html_data, parser)

# Find elements
title = doc.find('.//title').text
content_div = doc.find('.//div[@class="content"]')
list_items = doc.xpath('//li/text()')

print(f"Title: {title}")
print(f"List items: {list_items}")
```

### Error Handling

```python
from lxml import etree

try:
    # This will raise XMLSyntaxError due to unclosed tag
    bad_xml = '<root><child></root>'
    etree.fromstring(bad_xml)
except etree.XMLSyntaxError as e:
    print(f"XML Error: {e}")
    print(f"Line: {e.lineno}, Column: {e.offset}")

# Use recovery parser for malformed XML
try:
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(bad_xml, parser)
    print("Recovered:", etree.tostring(root, encoding='unicode'))
except Exception as e:
    print(f"Recovery failed: {e}")
```