# XPath and XSLT Processing

Advanced XML querying and transformation capabilities with XPath 1.0/2.0 evaluation, XSLT 1.0 stylesheets, extension functions, namespace handling, and XML canonicalization. These features enable powerful XML processing workflows for data extraction, transformation, and analysis.

## Capabilities

### XPath Evaluation

Compile and evaluate XPath expressions with variables, extension functions, and namespace support.

```python { .api }
class XPath:
    """Compiled XPath expression for efficient repeated evaluation."""
    
    def __init__(self, path, namespaces=None, extensions=None, 
                 regexp=True, smart_strings=True):
        """
        Compile XPath expression.
        
        Args:
            path: XPath expression string
            namespaces: dict mapping prefixes to namespace URIs
            extensions: dict of extension function modules
            regexp: Enable EXSLT regular expression functions
            smart_strings: Return Python str objects instead of lxml._ElementUnicodeResult objects
        """
    
    def __call__(self, _etree_or_element, **_variables):
        """
        Evaluate XPath on element or document.
        
        Args:
            _etree_or_element: Element or ElementTree to evaluate on
            **_variables: XPath variables as keyword arguments
        
        Returns:
            list: XPath evaluation results (elements, strings, numbers, or booleans depending on expression)
        """
    
    @property
    def path(self):
        """XPath expression string."""

class XPathEvaluator:
    """XPath evaluation context with persistent variables and functions."""
    
    def __init__(self, etree_or_element, namespaces=None, extensions=None,
                 enable_regexp=True, smart_strings=True):
        """
        Create XPath evaluator for specific element/document.
        
        Args:
            etree_or_element: Element or ElementTree to evaluate on
            namespaces: dict mapping prefixes to namespace URIs
            extensions: dict of extension function modules
            enable_regexp: Enable EXSLT regular expression functions
            smart_strings: Return Python str objects instead of lxml._ElementUnicodeResult objects
        """
    
    def __call__(self, _path, **_variables):
        """Evaluate XPath expression with variables."""
    
    def evaluate(self, _path, **_variables):
        """Evaluate XPath expression with variables."""
    
    def register_namespace(self, prefix, uri):
        """Register namespace prefix for this evaluator."""
    
    def register_namespaces(self, namespaces):
        """Register multiple namespace prefixes."""

class XPathDocumentEvaluator:
    """Document-level XPath evaluator with document context."""
    
    def __init__(self, etree, namespaces=None, extensions=None,
                 enable_regexp=True, smart_strings=True):
        """Create document-level XPath evaluator."""
    
    def __call__(self, _path, **_variables):
        """Evaluate XPath expression on document."""

# Element XPath methods
class Element:
    def xpath(self, _path, namespaces=None, extensions=None, 
              smart_strings=True, **_variables):
        """Evaluate XPath expression on element."""
```

### XSLT Transformation

Apply XSLT stylesheets to transform XML documents with parameters and extension functions.

```python { .api }
class XSLT:
    """XSLT stylesheet processor."""
    
    def __init__(self, xslt_input, extensions=None, regexp=True, 
                 access_control=None):
        """
        Create XSLT processor from stylesheet.
        
        Args:
            xslt_input: Element, ElementTree, or file containing XSLT
            extensions: dict of extension function modules
            regexp: Enable EXSLT regular expression functions
            access_control: XSLTAccessControl for security restrictions
        """
    
    def __call__(self, _input, profile_run=False, **kwargs):
        """
        Transform XML document using stylesheet.
        
        Args:
            _input: Element or ElementTree to transform
            profile_run: Enable XSLT profiling
            **kwargs: XSLT parameters as keyword arguments
        
        Returns:
            ElementTree: Transformation result
        """
    
    def apply(self, _input, **kwargs):
        """Apply transformation and return result tree."""
    
    def transform(self, _input, **kwargs):
        """Transform document (same as __call__)."""
    
    @property
    def error_log(self):
        """XSLT processing error log."""
    
    @staticmethod
    def strparam(s):
        """Convert Python string to XSLT string parameter."""

class XSLTAccessControl:
    """Security access control for XSLT processing to prevent unauthorized file/network access."""
    
    DENY_ALL = None        # Deny all external access (most secure)
    DENY_WRITE = None      # Deny write operations but allow reads
    DENY_READ = None       # Deny read operations but allow writes (rarely used)
    
    def __init__(self, read_file=True, write_file=False, create_dir=False,
                 read_network=False, write_network=False):
        """
        Create access control configuration for XSLT security.
        
        Args:
            read_file: Allow XSLT to read files from filesystem
            write_file: Allow XSLT to write files to filesystem (security risk)
            create_dir: Allow XSLT to create directories (security risk)
            read_network: Allow XSLT to fetch resources via HTTP/HTTPS (security risk)
            write_network: Allow XSLT to send data over network (security risk)
        """
```

### XML Canonicalization

XML canonicalization (C14N) for consistent XML representation and digital signatures.

```python { .api }
def canonicalize(xml_input, out=None, from_file=False, **options):
    """  
    Canonicalize XML document using C14N algorithm.
    
    Args:
        xml_input: XML string, Element, ElementTree, or filename
        out: Output file or file-like object (optional)
        from_file: Treat xml_input as filename
        **options: C14N options including:
            - exclusive: bool - Use exclusive canonicalization
            - with_comments: bool - Include comments (default True)  
            - inclusive_ns_prefixes: list - Namespace prefixes to include
            - strip_cdata: bool - Convert CDATA to text (default True)
    
    Returns:
        bytes: Canonicalized XML (if out not specified)
    """

class C14NWriterTarget:
    """Writer target for canonical XML output during parsing."""
    
    def __init__(self, write, **c14n_options):
        """
        Create C14N writer target.
        
        Args:
            write: Function to write canonicalized output
            **c14n_options: C14N canonicalization options
        """
```

### Extension Functions

Create custom XPath and XSLT extension functions.

```python { .api }
class Extension:
    """Base class for XSLT extensions."""

class XSLTExtension:
    """XSLT extension function handler."""

class FunctionNamespace:
    """XPath extension function namespace."""
    
    def __init__(self, namespace_uri):
        """
        Create function namespace.
        
        Args:
            namespace_uri: Namespace URI for extension functions
        """
    
    def __setitem__(self, function_name, function):
        """Register extension function."""
    
    def __getitem__(self, function_name):
        """Get registered extension function."""
    
    def __delitem__(self, function_name):
        """Unregister extension function."""
```

### XPath Error Handling

Comprehensive error classes for XPath and XSLT processing.

```python { .api }
class XPathError(LxmlError):
    """Base class for XPath-related errors."""

class XPathEvalError(XPathError):
    """XPath evaluation error."""

class XPathSyntaxError(XPathError):
    """XPath syntax error."""

class XPathResultError(XPathError):
    """XPath result type error."""

class XPathFunctionError(XPathError):
    """XPath function call error."""

class XSLTError(LxmlError):
    """Base class for XSLT-related errors."""

class XSLTParseError(XSLTError):
    """XSLT stylesheet parsing error."""

class XSLTApplyError(XSLTError):
    """XSLT transformation error."""

class XSLTSaveError(XSLTError):
    """XSLT result saving error."""

class XSLTExtensionError(XSLTError):
    """XSLT extension function error."""

class C14NError(LxmlError):
    """XML canonicalization error."""
```

## Usage Examples

### Basic XPath Queries

```python
from lxml import etree

# Sample XML document
xml_data = '''<?xml version="1.0"?>
<library xmlns:book="http://example.com/book">
    <book:catalog>
        <book:item id="1" category="fiction">
            <book:title>The Great Gatsby</book:title>
            <book:author>F. Scott Fitzgerald</book:author>
            <book:year>1925</book:year>
            <book:price currency="USD">12.99</book:price>
        </book:item>
        <book:item id="2" category="science">
            <book:title>A Brief History of Time</book:title>
            <book:author>Stephen Hawking</book:author>
            <book:year>1988</book:year>
            <book:price currency="USD">15.99</book:price>
        </book:item>
        <book:item id="3" category="fiction">
            <book:title>To Kill a Mockingbird</book:title>
            <book:author>Harper Lee</book:author>
            <book:year>1960</book:year>
            <book:price currency="USD">11.99</book:price>
        </book:item>
    </book:catalog>
</library>'''

root = etree.fromstring(xml_data)

# Define namespace mapping
namespaces = {'b': 'http://example.com/book'}

# Basic XPath queries
all_books = root.xpath('//b:item', namespaces=namespaces)
print(f"Found {len(all_books)} books")

fiction_books = root.xpath('//b:item[@category="fiction"]', namespaces=namespaces)
print(f"Fiction books: {len(fiction_books)}")

# Extract text content
titles = root.xpath('//b:title/text()', namespaces=namespaces)
print(f"Book titles: {titles}")

# Extract attributes
book_ids = root.xpath('//b:item/@id', namespaces=namespaces)
print(f"Book IDs: {book_ids}")

# Complex queries with predicates
expensive_books = root.xpath('//b:item[number(b:price) > 13]', namespaces=namespaces)
recent_books = root.xpath('//b:item[b:year > 1950]', namespaces=namespaces)

print(f"Expensive books: {len(expensive_books)}")
print(f"Recent books: {len(recent_books)}")

# XPath functions
oldest_book = root.xpath('//b:item[b:year = min(//b:year)]/b:title/text()', namespaces=namespaces)
print(f"Oldest book: {oldest_book[0] if oldest_book else 'None'}")
```

### Compiled XPath Expressions

```python
from lxml import etree

xml_data = '''
<products>
    <product id="1" price="19.99" category="electronics">
        <name>Widget</name>
        <stock>15</stock>
    </product>
    <product id="2" price="29.99" category="electronics">
        <name>Gadget</name>
        <stock>8</stock>
    </product>
    <product id="3" price="9.99" category="books">
        <name>Manual</name>
        <stock>25</stock>
    </product>
</products>
'''

root = etree.fromstring(xml_data)

# Compile XPath expressions for reuse
find_by_category = etree.XPath('//product[@category=$cat]')
find_by_price_range = etree.XPath('//product[number(@price) >= $min and number(@price) <= $max]')
count_in_stock = etree.XPath('sum(//product[@category=$cat]/stock)')

# Use compiled expressions with variables
electronics = find_by_category(root, cat='electronics')
print(f"Electronics products: {len(electronics)}")

affordable = find_by_price_range(root, min=10, max=25)
print(f"Affordable products: {len(affordable)}")

electronics_stock = count_in_stock(root, cat='electronics')
print(f"Total electronics in stock: {electronics_stock}")

# XPath evaluator for persistent context
evaluator = etree.XPathEvaluator(root)
evaluator.register_namespace('p', 'http://example.com/products')

# Evaluate multiple expressions with same context
product_count = evaluator('count(//product)')
avg_price = evaluator('sum(//product/@price) div count(//product)')
categories = evaluator('distinct-values(//product/@category)')

print(f"Products: {product_count}, Average price: ${avg_price:.2f}")
```

### XSLT Transformations

```python
from lxml import etree

# XML data to transform
xml_data = '''<?xml version="1.0"?>
<catalog>
    <book id="1">
        <title>Python Programming</title>
        <author>John Smith</author>
        <year>2023</year>
        <price>29.99</price>
    </book>
    <book id="2">
        <title>Web Development</title>
        <author>Jane Doe</author>
        <year>2022</year>
        <price>34.95</price>
    </book>
</catalog>'''

# XSLT stylesheet
xslt_stylesheet = '''<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:param name="format" select="'html'"/>
    <xsl:param name="title" select="'Book Catalog'"/>
    
    <xsl:template match="/">
        <xsl:choose>
            <xsl:when test="$format='html'">
                <html>
                    <head><title><xsl:value-of select="$title"/></title></head>
                    <body>
                        <h1><xsl:value-of select="$title"/></h1>
                        <table border="1">
                            <tr>
                                <th>Title</th>
                                <th>Author</th>
                                <th>Year</th>
                                <th>Price</th>
                            </tr>
                            <xsl:for-each select="catalog/book">
                                <xsl:sort select="year" order="descending"/>
                                <tr>
                                    <td><xsl:value-of select="title"/></td>
                                    <td><xsl:value-of select="author"/></td>
                                    <td><xsl:value-of select="year"/></td>
                                    <td>$<xsl:value-of select="price"/></td>
                                </tr>
                            </xsl:for-each>
                        </table>
                    </body>
                </html>
            </xsl:when>
            <xsl:otherwise>
                <book-list>
                    <xsl:for-each select="catalog/book">
                        <item>
                            <xsl:value-of select="title"/> by <xsl:value-of select="author"/> (<xsl:value-of select="year"/>)
                        </item>
                    </xsl:for-each>
                </book-list>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>
</xsl:stylesheet>'''

# Parse XML and XSLT
xml_doc = etree.fromstring(xml_data)
xslt_doc = etree.fromstring(xslt_stylesheet)

# Create XSLT processor
transform = etree.XSLT(xslt_doc)

# Transform with parameters
html_result = transform(xml_doc, format="'html'", title="'My Book Collection'")
print("HTML transformation:")
print(etree.tostring(html_result, pretty_print=True, encoding='unicode'))

# Transform with different parameters
text_result = transform(xml_doc, format="'text'")
print("\nText transformation:")
print(etree.tostring(text_result, pretty_print=True, encoding='unicode'))

# Check for transformation errors
if transform.error_log:
    print("XSLT errors:")
    for error in transform.error_log:
        print(f"  {error}")
```

### Extension Functions

```python
from lxml import etree

# Define custom extension functions
def custom_format_price(context, price_list, currency='USD'):
    """Format price with currency symbol."""
    if not price_list:
        return ''
    price = float(price_list[0])
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{price:.2f}"

def custom_word_count(context, text_list):
    """Count words in text."""
    if not text_list:
        return 0
    text = str(text_list[0])
    return len(text.split())

# Create extension namespace
ns = etree.FunctionNamespace('http://example.com/functions')
ns['format-price'] = custom_format_price
ns['word-count'] = custom_word_count

# XML with custom processing
xml_data = '''
<products>
    <product>
        <name>Programming Guide</name>
        <description>A comprehensive guide to Python programming for beginners and experts</description>
        <price>29.99</price>
    </product>
    <product>
        <name>Quick Reference</name>
        <description>Essential commands and functions</description>
        <price>15.50</price>
    </product>
</products>
'''

# XSLT using extension functions
xslt_with_extensions = '''<?xml version="1.0"?>
<xsl:stylesheet version="1.0" 
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:custom="http://example.com/functions">
    
    <xsl:template match="/">
        <product-report>
            <xsl:for-each select="products/product">
                <item>
                    <name><xsl:value-of select="name"/></name>
                    <formatted-price>
                        <xsl:value-of select="custom:format-price(price, 'USD')"/>
                    </formatted-price>
                    <description-length>
                        <xsl:value-of select="custom:word-count(description)"/> words
                    </description-length>
                </item>
            </xsl:for-each>
        </product-report>
    </xsl:template>
</xsl:stylesheet>
'''

# Transform using extensions
xml_doc = etree.fromstring(xml_data)
xslt_doc = etree.fromstring(xslt_with_extensions)

# Create transform with extensions enabled
extensions = {('http://example.com/functions', 'format-price'): custom_format_price,
              ('http://example.com/functions', 'word-count'): custom_word_count}

transform = etree.XSLT(xslt_doc, extensions=extensions)
result = transform(xml_doc)

print("Result with extension functions:")
print(etree.tostring(result, pretty_print=True, encoding='unicode'))
```

### XML Canonicalization

```python
from lxml import etree

# XML document with varying whitespace and attribute order
xml_data = '''<?xml version="1.0"?>
<root    xmlns:a="http://example.com/a"  
         xmlns:b="http://example.com/b">
    
    <element   b:attr="value2"   a:attr="value1"  >
        <child>   text content   </child>
        <!-- This is a comment -->
        <another-child/>
    </element>
    
</root>'''

# Parse document
doc = etree.fromstring(xml_data)

# Basic canonicalization
canonical_xml = etree.canonicalize(xml_data)
print("Canonical XML (default):")
print(canonical_xml.decode('utf-8'))

# Canonicalization without comments
canonical_no_comments = etree.canonicalize(xml_data, with_comments=False)
print("\nCanonical XML (no comments):")
print(canonical_no_comments.decode('utf-8'))

# Exclusive canonicalization
canonical_exclusive = etree.canonicalize(xml_data, exclusive=True)
print("\nExclusive canonical XML:")
print(canonical_exclusive.decode('utf-8'))

# Canonicalize to file
with open('/tmp/canonical.xml', 'wb') as f:
    etree.canonicalize(xml_data, out=f)

# Using C14N writer target during parsing
output_parts = []
def write_canonical(data):
    output_parts.append(data)

target = etree.C14NWriterTarget(write_canonical, with_comments=False)
parser = etree.XMLParser(target=target)
etree.fromstring(xml_data, parser)

print("\nCanonical XML via writer target:")
print(b''.join(output_parts).decode('utf-8'))
```

### Advanced XPath with Namespaces

```python
from lxml import etree

# Complex XML with multiple namespaces
xml_data = '''<?xml version="1.0"?>
<root xmlns="http://example.com/default"
      xmlns:meta="http://example.com/metadata"
      xmlns:content="http://example.com/content">
    
    <meta:info>
        <meta:created>2023-12-07</meta:created>
        <meta:author>John Doe</meta:author>
    </meta:info>
    
    <content:document>
        <content:section id="intro">
            <content:title>Introduction</content:title>
            <content:paragraph>This is the introduction.</content:paragraph>
        </content:section>
        <content:section id="main">
            <content:title>Main Content</content:title>
            <content:paragraph>This is the main content.</content:paragraph>
            <content:subsection>
                <content:title>Subsection</content:title>
                <content:paragraph>Subsection content.</content:paragraph>
            </content:subsection>
        </content:section>
    </content:document>
    
</root>'''

root = etree.fromstring(xml_data)

# Define comprehensive namespace mappings
namespaces = {
    'default': 'http://example.com/default',
    'meta': 'http://example.com/metadata', 
    'content': 'http://example.com/content'
}

# Complex XPath queries with namespaces
author = root.xpath('//meta:author/text()', namespaces=namespaces)
print(f"Author: {author[0] if author else 'Unknown'}")

# Find all sections and subsections
sections = root.xpath('//content:section | //content:subsection', namespaces=namespaces)
print(f"Found {len(sections)} sections")

# Extract titles with context
titles_with_id = root.xpath('//content:section[@id]/content:title/text()', namespaces=namespaces)
for title in titles_with_id:
    print(f"Section title: {title}")

# Count paragraphs in main section
main_paragraphs = root.xpath('count(//content:section[@id="main"]//content:paragraph)', namespaces=namespaces)
print(f"Paragraphs in main section: {main_paragraphs}")

# Build document outline
outline_xpath = etree.XPath('''
    for $section in //content:section
    return concat($section/@id, ": ", $section/content:title/text())
''', namespaces=namespaces)

outline = outline_xpath(root)
print("Document outline:")
for item in outline:
    print(f"  {item}")
```