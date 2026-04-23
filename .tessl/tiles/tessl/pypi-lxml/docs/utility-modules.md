# Utility Modules

Additional functionality including SAX interface compatibility, CSS selector support, element builders, XInclude processing, and namespace management. These modules provide specialized capabilities for integration with other XML tools and advanced XML processing workflows.

## Capabilities

### SAX Interface Compatibility

Bridge between lxml and Python's SAX (Simple API for XML) for integration with SAX-based applications.

```python { .api }
class ElementTreeContentHandler:
    """SAX ContentHandler that builds lxml ElementTree."""
    
    def __init__(self, makeelement=None):
        """
        Create SAX content handler for building ElementTree.
        
        Args:
            makeelement: Custom element factory function (optional)
        """
    
    def etree(self):
        """Get built ElementTree after parsing completes."""
    
    # SAX ContentHandler interface methods
    def setDocumentLocator(self, locator): ...
    def startDocument(self): ...
    def endDocument(self): ...
    def startPrefixMapping(self, prefix, uri): ...
    def endPrefixMapping(self, prefix): ...
    def startElement(self, name, attrs): ...
    def endElement(self, name): ...
    def startElementNS(self, name, qname, attrs): ...
    def endElementNS(self, name, qname): ...
    def characters(self, data): ...
    def ignorableWhitespace(self, whitespace): ...
    def processingInstruction(self, target, data): ...
    def skippedEntity(self, name): ...

class ElementTreeProducer:
    """Generate SAX events from lxml ElementTree."""
    
    def __init__(self, element_or_tree, content_handler):
        """
        Create SAX event producer.
        
        Args:
            element_or_tree: Element or ElementTree to process
            content_handler: SAX ContentHandler to receive events
        """
    
    def saxify(self):
        """Generate SAX events for the element tree."""

def saxify(element_or_tree, content_handler):
    """
    Generate SAX events from lxml tree.
    
    Args:
        element_or_tree: Element or ElementTree to process
        content_handler: SAX ContentHandler to receive events
    """

class SaxError(LxmlError):
    """SAX processing error."""
```

### CSS Selectors

CSS selector support for finding elements using CSS syntax instead of XPath.

```python { .api }
class CSSSelector:
    """CSS selector that compiles to XPath for element matching."""
    
    def __init__(self, css, namespaces=None, translator='xml'):
        """
        Create CSS selector.
        
        Args:
            css: CSS selector string
            namespaces: dict mapping prefixes to namespace URIs
            translator: Selector translator ('xml' or 'html')
        """
    
    def __call__(self, element):
        """
        Find elements matching CSS selector.
        
        Args:
            element: Element or ElementTree to search
        
        Returns:
            list: Matching elements
        """
    
    @property
    def css(self):
        """CSS selector string."""
    
    @property
    def path(self):
        """Compiled XPath expression."""

class LxmlTranslator:
    """CSS to XPath translator with lxml-specific extensions."""
    
    def css_to_xpath(self, css, prefix='descendant-or-self::'):
        """Convert CSS selector to XPath expression."""

class LxmlHTMLTranslator(LxmlTranslator):
    """HTML-specific CSS to XPath translator."""

# CSS selector error classes
class SelectorSyntaxError(Exception):
    """CSS selector syntax error."""

class ExpressionError(Exception):
    """CSS expression error."""

class SelectorError(Exception):
    """General CSS selector error."""
```

### Element Builders

Factory classes for programmatically creating XML elements with fluent APIs.

```python { .api }
class ElementMaker:
    """Factory for creating XML elements with builder pattern."""
    
    def __init__(self, typemap=None, namespace=None, nsmap=None, 
                 makeelement=None, **default_attributes):
        """
        Create element factory.
        
        Args:
            typemap: dict mapping Python types to conversion functions
            namespace: Default namespace URI for created elements
            nsmap: Namespace prefix mapping
            makeelement: Custom element factory function
            **default_attributes: Default attributes for all elements
        """
    
    def __call__(self, tag, *children, **attributes):
        """
        Create element with tag, children, and attributes.
        
        Args:
            tag: Element tag name
            *children: Child elements, text, or other content
            **attributes: Element attributes
        
        Returns:
            Element: Created element with children and attributes
        """
    
    def __getattr__(self, tag):
        """Create element factory method for specific tag."""

# Default element maker instance
E = ElementMaker()
```

### XInclude Processing

XML Inclusions (XInclude) processing for modular XML documents.

```python { .api }
def include(elem, loader=None, base_url=None, max_depth=6):
    """
    Process XInclude directives in element tree.
    
    Args:
        elem: Element containing XInclude directives
        loader: Custom resource loader function
        base_url: Base URL for resolving relative hrefs
        max_depth: Maximum inclusion recursion depth
    
    Raises:
        FatalIncludeError: Fatal inclusion error
        LimitedRecursiveIncludeError: Recursion limit exceeded
    """

def default_loader(href, parse, encoding=None):
    """
    Default XInclude resource loader.
    
    Args:
        href: Resource URI to load
        parse: Parse mode ('xml' or 'text')
        encoding: Character encoding for text resources
    
    Returns:
        Element or str: Loaded resource content
    """

class FatalIncludeError(LxmlError):
    """Fatal XInclude processing error."""

class LimitedRecursiveIncludeError(FatalIncludeError):
    """XInclude recursion limit exceeded."""

# XInclude constants
DEFAULT_MAX_INCLUSION_DEPTH = 6
XINCLUDE_NAMESPACE = "http://www.w3.org/2001/XInclude"
```

### ElementPath Support

Simple XPath-like expressions for element tree navigation (similar to ElementTree).

```python { .api }
def find(element, path, namespaces=None):
    """
    Find first element matching simple path expression.
    
    Args:
        element: Element to search from
        path: Simple path expression (e.g., 'child/grandchild')
        namespaces: Namespace prefix mapping
    
    Returns:
        Element or None: First matching element
    """

def findall(element, path, namespaces=None):
    """
    Find all elements matching simple path expression.
    
    Args:
        element: Element to search from
        path: Simple path expression
        namespaces: Namespace prefix mapping
    
    Returns:
        list: All matching elements
    """

def iterfind(element, path, namespaces=None):
    """
    Iterate over elements matching simple path expression.
    
    Args:
        element: Element to search from
        path: Simple path expression
        namespaces: Namespace prefix mapping
    
    Yields:
        Element: Matching elements
    """

def findtext(element, path, default=None, namespaces=None):
    """
    Find text content of first element matching path.
    
    Args:
        element: Element to search from
        path: Simple path expression
        default: Default value if no match found
        namespaces: Namespace prefix mapping
    
    Returns:
        str or default: Text content or default value
    """
```

### Document Testing Utilities

Enhanced utilities for testing XML documents and doctests.

```python { .api }
# lxml.usedoctest - doctest support
def temp_install(modules=None, verbose=None):
    """Temporarily install lxml doctests."""

# lxml.doctestcompare - enhanced doctest comparison  
class LXMLOutputChecker:
    """Enhanced output checker for XML doctests."""
    
    def check_output(self, want, got, optionflags):
        """Compare expected and actual XML output."""

class LHTMLOutputChecker:
    """Enhanced output checker for HTML doctests."""

# Test options
PARSE_HTML = ...
PARSE_XML = ...
NOPARSE_MARKUP = ...
```

### Python Class Lookup

Custom element class assignment based on Python logic.

```python { .api }
# lxml.pyclasslookup - Python-based element class lookup
class PythonElementClassLookup:
    """Element class lookup using Python callback functions."""
    
    def __init__(self, fallback=None):
        """
        Create Python-based class lookup.
        
        Args:
            fallback: Fallback class lookup for unhandled cases
        """
    
    def lookup(self, doc, element):
        """
        Lookup element class based on document and element.
        
        Args:
            doc: Document containing element
            element: Element to assign class for
        
        Returns:
            type or None: Element class or None for default
        """
```

### Development Utilities

Helper functions for development and compilation workflows.

```python { .api }
def get_include():
    """
    Returns header include paths for compiling C code against lxml.
    
    Returns paths for lxml itself, libxml2, and libxslt headers when lxml
    was built with statically linked libraries.
    
    Returns:
        list: List of include directory paths
    """
```

## Usage Examples

### SAX Interface Integration

```python
from lxml import etree
from lxml.sax import ElementTreeContentHandler, saxify
from xml.sax import make_parser
import xml.sax.handler

# Build ElementTree from SAX events
class MyContentHandler(ElementTreeContentHandler):
    def __init__(self):
        super().__init__()
        self.elements_seen = []
    
    def startElement(self, name, attrs):
        super().startElement(name, attrs)
        self.elements_seen.append(name)

# Parse XML using SAX, build with lxml
xml_data = '''<?xml version="1.0"?>
<catalog>
    <book id="1">
        <title>Python Guide</title>
        <author>John Doe</author>
    </book>
    <book id="2">
        <title>XML Processing</title>
        <author>Jane Smith</author>
    </book>
</catalog>'''

handler = MyContentHandler()
parser = make_parser()
parser.setContentHandler(handler)

# Parse and get resulting ElementTree
from io import StringIO
parser.parse(StringIO(xml_data))
tree = handler.etree()

print(f"Elements seen: {handler.elements_seen}")
print(f"Root tag: {tree.getroot().tag}")

# Generate SAX events from lxml tree
class LoggingHandler(xml.sax.handler.ContentHandler):
    def startElement(self, name, attrs):
        print(f"Start: {name} {dict(attrs)}")
    
    def endElement(self, name):
        print(f"End: {name}")
    
    def characters(self, content):
        content = content.strip()
        if content:
            print(f"Text: {content}")

# Send lxml tree to SAX handler
root = etree.fromstring(xml_data)
logging_handler = LoggingHandler()
saxify(root, logging_handler)
```

### CSS Selectors

```python
from lxml import html
from lxml.cssselect import CSSSelector

# HTML document for CSS selection
html_content = '''
<html>
<head>
    <title>CSS Selector Example</title>
</head>
<body>
    <div id="header" class="main-header">
        <h1>Welcome</h1>
        <nav class="navigation">
            <a href="/home" class="nav-link active">Home</a>
            <a href="/about" class="nav-link">About</a>
            <a href="/contact" class="nav-link">Contact</a>
        </nav>
    </div>
    <div id="content" class="main-content">
        <article class="post featured">
            <h2>Featured Article</h2>
            <p>This is a featured article.</p>
        </article>
        <article class="post">
            <h2>Regular Article</h2>
            <p>This is a regular article.</p>
        </article>
    </div>
    <footer id="footer">
        <p>&copy; 2023 Example Site</p>
    </footer>
</body>
</html>
'''

doc = html.fromstring(html_content)

# Create CSS selectors
header_selector = CSSSelector('#header')
nav_links_selector = CSSSelector('nav.navigation a.nav-link')
featured_post_selector = CSSSelector('article.post.featured')
all_headings_selector = CSSSelector('h1, h2, h3, h4, h5, h6')

# Use selectors to find elements
header = header_selector(doc)
print(f"Header element: {header[0].get('class') if header else 'Not found'}")

nav_links = nav_links_selector(doc)
print(f"Navigation links: {len(nav_links)}")
for link in nav_links:
    print(f"  {link.text}: {link.get('href')}")

featured = featured_post_selector(doc)
if featured:
    print(f"Featured article title: {featured[0].find('.//h2').text}")

headings = all_headings_selector(doc)
print(f"All headings:")
for heading in headings:
    print(f"  {heading.tag}: {heading.text}")

# Advanced CSS selectors
active_link_selector = CSSSelector('a.nav-link.active')
first_paragraph_selector = CSSSelector('article p:first-child')
not_featured_selector = CSSSelector('article.post:not(.featured)')

active_links = active_link_selector(doc)
print(f"Active navigation links: {len(active_links)}")

first_paragraphs = first_paragraph_selector(doc)
print(f"First paragraphs in articles: {len(first_paragraphs)}")

regular_posts = not_featured_selector(doc)
print(f"Regular (non-featured) posts: {len(regular_posts)}")
```

### Element Builders

```python
from lxml import etree
from lxml.builder import ElementMaker

# Create element maker with namespace
E = ElementMaker(namespace="http://example.com/catalog",
                 nsmap={None: "http://example.com/catalog"})

# Build XML structure using element maker
catalog = E.catalog(
    E.metadata(
        E.title("Book Catalog"),
        E.created("2023-12-07"),
        E.version("1.0")
    ),
    E.books(
        E.book(
            E.title("Python Programming"),
            E.author("John Smith"),
            E.isbn("978-0123456789"),
            E.price("29.99", currency="USD"),
            E.categories(
                E.category("Programming"),
                E.category("Python"),
                E.category("Computers")
            ),
            id="1",
            available="true"
        ),
        E.book(
            E.title("Web Development"),
            E.author("Jane Doe"),
            E.isbn("978-0987654321"),
            E.price("34.95", currency="USD"),
            E.categories(
                E.category("Web"),
                E.category("HTML"),
                E.category("CSS")
            ),
            id="2",
            available="false"
        )
    )
)

print("Generated XML:")
print(etree.tostring(catalog, pretty_print=True, encoding='unicode'))

# Custom element maker with type mapping
def format_price(value):
    """Custom price formatter."""
    return f"${float(value):.2f}"

def bool_to_string(value):
    """Convert boolean to string."""
    return "yes" if value else "no"

custom_typemap = {
    float: format_price,
    bool: bool_to_string
}

CustomE = ElementMaker(typemap=custom_typemap)

# Use custom element maker
product = CustomE.product(
    CustomE.name("Widget"),
    CustomE.price(19.99),  # Will be formatted as currency
    CustomE.available(True),  # Will be converted to "yes"
    CustomE.features(
        CustomE.feature("Lightweight"),
        CustomE.feature("Durable"),
        CustomE.feature("Affordable")
    )
)

print("\nCustom formatted XML:")
print(etree.tostring(product, pretty_print=True, encoding='unicode'))
```

### XInclude Processing

```python
from lxml import etree
from lxml.ElementInclude import include, default_loader
import tempfile
import os

# Create temporary files for XInclude example
temp_dir = tempfile.mkdtemp()

# Create included content files
header_content = '''<?xml version="1.0"?>
<header>
    <title>Document Title</title>
    <author>John Doe</author>
    <date>2023-12-07</date>
</header>'''

footer_content = '''<?xml version="1.0"?>
<footer>
    <copyright>&copy; 2023 Example Corp</copyright>
    <contact>contact@example.com</contact>
</footer>'''

# Write include files
header_file = os.path.join(temp_dir, 'header.xml')
footer_file = os.path.join(temp_dir, 'footer.xml')

with open(header_file, 'w') as f:
    f.write(header_content)

with open(footer_file, 'w') as f:
    f.write(footer_content)

# Main document with XInclude directives
main_doc_content = f'''<?xml version="1.0"?>
<document xmlns:xi="http://www.w3.org/2001/XInclude">
    <xi:include href="{header_file}"/>
    
    <content>
        <section>
            <h1>Introduction</h1>
            <p>This is the main content of the document.</p>
        </section>
        <section>
            <h1>Details</h1>
            <p>More detailed information goes here.</p>
        </section>
    </content>
    
    <xi:include href="{footer_file}"/>
</document>'''

# Parse document with XInclude processing
root = etree.fromstring(main_doc_content)
print("Before XInclude processing:")
print(etree.tostring(root, pretty_print=True, encoding='unicode'))

# Process XInclude directives
include(root)
print("\nAfter XInclude processing:")
print(etree.tostring(root, pretty_print=True, encoding='unicode'))

# Custom loader for XInclude
def custom_loader(href, parse, encoding=None):
    """Custom XInclude loader with logging."""
    print(f"Loading: {href} (parse={parse}, encoding={encoding})")
    return default_loader(href, parse, encoding)

# Use custom loader
root2 = etree.fromstring(main_doc_content)
include(root2, loader=custom_loader)

# Clean up temporary files
os.unlink(header_file)
os.unlink(footer_file)
os.rmdir(temp_dir)
```

### ElementPath Simple Queries

```python
from lxml import etree
from lxml._elementpath import find, findall, iterfind, findtext

# XML document for path queries
xml_data = '''<?xml version="1.0"?>
<library>
    <section name="fiction">
        <book id="1">
            <title>The Great Gatsby</title>
            <author>F. Scott Fitzgerald</author>
            <metadata>
                <genre>Classic Literature</genre>
                <year>1925</year>
            </metadata>
        </book>
        <book id="2">
            <title>To Kill a Mockingbird</title>
            <author>Harper Lee</author>
            <metadata>
                <genre>Classic Literature</genre>
                <year>1960</year>
            </metadata>
        </book>
    </section>
    <section name="science">
        <book id="3">
            <title>A Brief History of Time</title>
            <author>Stephen Hawking</author>
            <metadata>
                <genre>Science</genre>
                <year>1988</year>
            </metadata>
        </book>
    </section>
</library>'''

root = etree.fromstring(xml_data)

# Simple path queries (ElementTree-style)
fiction_section = find(root, 'section[@name="fiction"]')
print(f"Fiction section: {fiction_section.get('name') if fiction_section else 'Not found'}")

# Find all books in any section
all_books = findall(root, './/book')
print(f"Total books: {len(all_books)}")

# Find specific book by ID
book1 = find(root, './/book[@id="1"]')
if book1:
    title = findtext(book1, 'title')
    author = findtext(book1, 'author') 
    print(f"Book 1: {title} by {author}")

# Iterate over books in fiction section
fiction_books = iterfind(root, 'section[@name="fiction"]/book')
print("Fiction books:")
for book in fiction_books:
    title = findtext(book, 'title')
    year = findtext(book, 'metadata/year')
    print(f"  {title} ({year})")

# Find text with default value
unknown_book = findtext(root, 'section/book[@id="999"]/title', 'Unknown Book')
print(f"Unknown book title: {unknown_book}")

# Complex paths
classic_books = findall(root, './/book[metadata/genre="Classic Literature"]')
print(f"Classic literature books: {len(classic_books)}")

recent_books = findall(root, './/book[metadata/year>"1950"]')
print(f"Books after 1950: {len(recent_books)}")
```

### Custom Element Classes

```python
from lxml import etree

# Define custom element classes
class BookElement(etree.ElementBase):
    """Custom element class for book elements."""
    
    @property
    def title(self):
        """Get book title."""
        title_elem = self.find('title')
        return title_elem.text if title_elem is not None else None
    
    @property
    def author(self):
        """Get book author."""
        author_elem = self.find('author')
        return author_elem.text if author_elem is not None else None
    
    @property
    def year(self):
        """Get publication year as integer."""
        year_elem = self.find('metadata/year')
        if year_elem is not None:
            try:
                return int(year_elem.text)
            except (ValueError, TypeError):
                return None
        return None
    
    def is_classic(self):
        """Check if book is classic literature."""
        genre_elem = self.find('metadata/genre')
        return genre_elem is not None and genre_elem.text == 'Classic Literature'

class SectionElement(etree.ElementBase):
    """Custom element class for section elements."""
    
    @property
    def name(self):
        """Get section name."""
        return self.get('name', 'Unnamed Section')
    
    def get_books(self):
        """Get all books in this section."""
        return self.findall('book')
    
    def count_books(self):
        """Count books in this section."""
        return len(self.findall('book'))

# Create element class lookup
class CustomElementClassLookup(etree.PythonElementClassLookup):
    def lookup(self, document, element):
        if element.tag == 'book':
            return BookElement
        elif element.tag == 'section':
            return SectionElement
        return None

# Set up parser with custom lookup
lookup = CustomElementClassLookup()
parser = etree.XMLParser()
parser.set_element_class_lookup(lookup)

# Parse with custom element classes
xml_data = '''<?xml version="1.0"?>
<library>
    <section name="fiction">
        <book id="1">
            <title>The Great Gatsby</title>
            <author>F. Scott Fitzgerald</author>
            <metadata>
                <genre>Classic Literature</genre>
                <year>1925</year>
            </metadata>
        </book>
        <book id="2">
            <title>Modern Fiction</title>
            <author>Contemporary Author</author>
            <metadata>
                <genre>Modern Literature</genre>
                <year>2020</year>
            </metadata>
        </book>
    </section>
</library>'''

root = etree.fromstring(xml_data, parser)

# Use custom element methods
fiction_section = root.find('section')
print(f"Section: {fiction_section.name}")
print(f"Books in section: {fiction_section.count_books()}")

for book in fiction_section.get_books():
    print(f"  {book.title} by {book.author} ({book.year})")
    print(f"    Is classic: {book.is_classic()}")
```