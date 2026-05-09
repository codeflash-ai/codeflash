# lxml

A comprehensive Python library for processing XML and HTML documents. lxml combines the speed and feature completeness of libxml2 and libxslt with the simplicity of Python's ElementTree API, providing fast, standards-compliant XML/HTML processing with extensive validation, transformation, and manipulation capabilities.

## Package Information

- **Package Name**: lxml
- **Language**: Python
- **Installation**: `pip install lxml`
- **Documentation**: https://lxml.de/
- **Requirements**: Python 3.8+

## Core Imports

The library provides multiple APIs optimized for different use cases:

```python
# Core XML/HTML processing (ElementTree-compatible)
from lxml import etree

# Object-oriented XML API with Python data type mapping
from lxml import objectify

# HTML-specific processing with form/link handling
from lxml import html

# Schema validation
from lxml.isoschematron import Schematron

# CSS selector support
from lxml.cssselect import CSSSelector
```

## Basic Usage

### XML Processing

```python
from lxml import etree

# Parse XML from string
xml_data = """
<bookstore>
    <book id="1">
        <title>Python Guide</title>
        <author>Jane Smith</author>
        <price>29.99</price>
    </book>
    <book id="2">
        <title>XML Processing</title>
        <author>John Doe</author>
        <price>34.95</price>
    </book>
</bookstore>
"""

root = etree.fromstring(xml_data)

# Find elements using XPath
books = root.xpath('//book[@id="1"]')
print(books[0].find('title').text)  # "Python Guide"

# Create new elements
new_book = etree.SubElement(root, 'book', id="3")
etree.SubElement(new_book, 'title').text = "Advanced Topics"
etree.SubElement(new_book, 'author').text = "Alice Johnson"
etree.SubElement(new_book, 'price').text = "39.99"

# Serialize back to XML
print(etree.tostring(root, pretty_print=True, encoding='unicode'))
```

### HTML Processing

```python
from lxml import html

# Parse HTML
html_content = """
<html>
<head><title>Example Page</title></head>
<body>
    <form action="/submit" method="post">
        <input type="text" name="username" value="john">
        <input type="password" name="password">
        <button type="submit">Login</button>
    </form>
    <div class="content">
        <a href="https://example.com">External Link</a>
        <a href="/internal">Internal Link</a>
    </div>
</body>
</html>
"""

doc = html.fromstring(html_content)

# Find form elements
form = doc.forms[0]
print(form.fields)  # Form field dictionary

# Process links
html.make_links_absolute(doc, base_url='https://mysite.com')
for element, attribute, link, pos in html.iterlinks(doc):
    print(f"{element.tag}.{attribute}: {link}")
```

### Object-Oriented API

```python
from lxml import objectify

# Parse XML into Python objects
xml_data = """
<data>
    <items>
        <item>
            <name>Widget</name>
            <price>19.99</price>
            <available>true</available>
        </item>
    </items>
</data>
"""

root = objectify.fromstring(xml_data)

# Access as Python attributes
print(root.items.item.name)      # "Widget"
print(root.items.item.price)     # 19.99 (automatically converted to float)
print(root.items.item.available) # True (automatically converted to bool)

# Add new data
root.items.item.category = "Electronics"
print(objectify.dump(root))
```

## Architecture

lxml provides multiple complementary APIs built on a common foundation:

- **etree**: Low-level ElementTree-compatible API for precise XML/HTML control
- **objectify**: High-level Pythonic API with automatic type conversion  
- **html**: Specialized HTML processing with web-specific features
- **Validation**: Multiple schema languages (DTD, RelaxNG, XML Schema, Schematron)
- **Processing**: XPath queries, XSLT transformations, canonicalization

The library's Cython implementation provides C-level performance while maintaining Python's ease of use, making it suitable for both simple scripts and high-performance applications processing large XML documents.

## Capabilities

### Core XML/HTML Processing

Low-level ElementTree-compatible API providing comprehensive XML and HTML parsing, manipulation, and serialization with full namespace support, error handling, and memory-efficient processing.

```python { .api }
# Parsing functions
def parse(source, parser=None, base_url=None): ...
def fromstring(text, parser=None, base_url=None): ...
def XML(text, parser=None, base_url=None): ...
def HTML(text, parser=None, base_url=None): ...

# Core classes
class Element: ...
class ElementTree: ...
class XMLParser: ...
class HTMLParser: ...

# Serialization
def tostring(element_or_tree, encoding=None, method='xml', pretty_print=False): ...
```

[Core XML/HTML Processing](./etree-core.md)

### Object-Oriented XML API

Pythonic XML processing that automatically converts XML data to Python objects with proper data types, providing intuitive attribute-based access and manipulation while maintaining full XML structure.

```python { .api }
# Parsing functions
def parse(source, parser=None, base_url=None): ...
def fromstring(text, parser=None, base_url=None): ...

# Core classes
class ObjectifiedElement: ...
class DataElement: ...
class ElementMaker: ...

# Type annotation functions
def annotate(element_or_tree, **kwargs): ...
def deannotate(element_or_tree, **kwargs): ...
```

[Object-Oriented XML API](./objectify-api.md)

### HTML Processing

Specialized HTML document processing with web-specific features including form handling, link processing, CSS class manipulation, and HTML5 parsing support.

```python { .api }
# HTML parsing
def parse(filename_or_url, parser=None, base_url=None): ...
def fromstring(html, base_url=None, parser=None): ...
def document_fromstring(html, parser=None, ensure_head_body=False): ...

# Link processing
def make_links_absolute(element, base_url=None): ...
def iterlinks(element): ...
def rewrite_links(element, link_repl_func): ...

# Form handling
def submit_form(form, extra_values=None, open_http=None): ...
```

[HTML Processing](./html-processing.md)

### Schema Validation

Comprehensive XML validation support including DTD, RelaxNG, W3C XML Schema, and ISO Schematron with detailed error reporting and custom validation rules.

```python { .api }
class DTD: ...
class RelaxNG: ...
class XMLSchema: ...

# Schematron validation
class Schematron: ...
def extract_xsd(element): ...
def extract_rng(element): ...
```

[Schema Validation](./validation.md)

### XPath and XSLT Processing

Advanced XML querying and transformation capabilities with XPath 1.0/2.0 evaluation, XSLT 1.0 stylesheets, extension functions, and namespace handling.

```python { .api }
class XPath: ...
class XPathEvaluator: ...
class XSLT: ...
class XSLTAccessControl: ...

# Utility functions
def canonicalize(xml_data, **options): ...
```

[XPath and XSLT Processing](./xpath-xslt.md)

### Utility Modules

Additional functionality including SAX interface compatibility, CSS selector support, element builders, XInclude processing, and namespace management.

```python { .api }
# SAX interface
class ElementTreeContentHandler: ...
def saxify(element_or_tree, content_handler): ...

# CSS selectors
class CSSSelector: ...

# Element builders
class ElementMaker: ...

# Development utilities
def get_include(): ...
```

[Utility Modules](./utility-modules.md)

## Error Handling

lxml provides a comprehensive exception hierarchy for precise error handling:

```python { .api }
class LxmlError(Exception): ...
class XMLSyntaxError(LxmlError): ...
class DTDError(LxmlError): ...
class RelaxNGError(LxmlError): ...
class XMLSchemaError(LxmlError): ...
class XPathError(LxmlError): ...
class XSLTError(LxmlError): ...
```

All validation and processing functions raise specific exceptions with detailed error messages and line number information when available.

## Types

### Core Types

```python { .api }
class Element:
    """XML element with tag, attributes, text content, and children."""
    tag: str
    text: str | None
    tail: str | None
    attrib: dict[str, str]
    
    def find(self, path: str, namespaces: dict[str, str] = None) -> Element | None: ...
    def findall(self, path: str, namespaces: dict[str, str] = None) -> list[Element]: ...
    def xpath(self, path: str, **kwargs) -> list: ...
    def get(self, key: str, default: str = None) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...

class ElementTree:
    """Document tree with root element and document-level operations."""
    def getroot(self) -> Element: ...
    def write(self, file, encoding: str = None, xml_declaration: bool = None): ...
    def xpath(self, path: str, **kwargs) -> list: ...

class QName:
    """Qualified name with namespace URI and local name."""
    def __init__(self, text_or_uri_or_element, tag: str = None): ...
    localname: str
    namespace: str | None
    text: str
```

### Parser Types

```python { .api }
class XMLParser:
    """Configurable XML parser with validation and error handling options."""
    def __init__(self, encoding: str = None, remove_blank_text: bool = False, 
                 remove_comments: bool = False, remove_pis: bool = False,
                 strip_cdata: bool = True, recover: bool = False, **kwargs): ...

class HTMLParser:
    """Lenient HTML parser with automatic error recovery."""
    def __init__(self, encoding: str = None, remove_blank_text: bool = False,
                 remove_comments: bool = False, **kwargs): ...

ParserType = XMLParser | HTMLParser
```