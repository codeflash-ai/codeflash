# Object-Oriented XML API

Pythonic XML processing that automatically converts XML elements to native Python objects with proper data types. The objectify module provides intuitive attribute-based access to XML content while maintaining full XML structure and namespace support.

## Capabilities

### Document Parsing

Parse XML documents into objectified trees with automatic type conversion.

```python { .api }
def parse(source, parser=None, base_url=None):
    """
    Parse XML document into objectified tree.
    
    Args:
        source: File path, URL, or file-like object
        parser: ObjectifyElementClassLookup-enabled parser (optional)
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        ObjectifiedElement: Root element with objectified children
    """

def fromstring(text, parser=None, base_url=None):
    """
    Parse XML string into objectified element.
    
    Args:
        text: str or bytes containing XML content
        parser: ObjectifyElementClassLookup-enabled parser (optional) 
        base_url: Base URL for resolving relative references (optional)
    
    Returns:
        ObjectifiedElement: Root element with objectified children
    """

def XML(text, parser=None, base_url=None):
    """Parse XML string into objectified element with validation."""
```

### Objectified Elements

Elements that provide Python object-like access to XML content with automatic type conversion.

```python { .api }
class ObjectifiedElement:
    """
    XML element with Python object-like attribute access and type conversion.
    """
    
    # Attribute access - returns child elements or text content
    def __getattr__(self, name):
        """Access child elements as attributes."""
    
    def __setattr__(self, name, value):
        """Set child element content or create new elements."""
    
    def __delattr__(self, name):
        """Remove child elements."""
    
    # List-like access for multiple children with same name
    def __getitem__(self, index):
        """Access children by index."""
    
    def __len__(self):
        """Number of child elements."""
    
    def __iter__(self):
        """Iterate over child elements."""
    
    # Dictionary-like access for attributes
    def get(self, key, default=None):
        """Get XML attribute value."""
    
    def set(self, key, value):
        """Set XML attribute value."""
    
    # Python value conversion
    def __str__(self):
        """String representation of element text content."""
    
    def __int__(self):
        """Integer conversion of element text."""
    
    def __float__(self):
        """Float conversion of element text."""
    
    def __bool__(self):
        """Boolean conversion of element text."""
    
    # XML methods (inherited from Element)
    def xpath(self, path, **kwargs):
        """XPath evaluation on objectified tree."""
    
    def find(self, path, namespaces=None):
        """Find child elements."""
    
    def findall(self, path, namespaces=None):
        """Find all child elements."""

class ObjectifiedDataElement:
    """Objectified element containing typed data."""
    
    @property 
    def pyval(self):
        """Python value with automatic type conversion."""
    
    @pyval.setter
    def pyval(self, value):
        """Set Python value with type annotation."""
```

### Data Element Classes

Specialized element classes for different Python data types.

```python { .api }
class StringElement(ObjectifiedDataElement):
    """Element containing string data."""

class IntElement(ObjectifiedDataElement):
    """Element containing integer data."""

class FloatElement(ObjectifiedDataElement):
    """Element containing floating-point data."""

class BoolElement(ObjectifiedDataElement):
    """Element containing boolean data."""

class NoneElement(ObjectifiedDataElement):
    """Element representing None/null value."""

class NumberElement(ObjectifiedDataElement):
    """Element containing numeric data (int or float auto-detected)."""

def DataElement(_value, _pytype=None, _xsi=None, **kwargs):
    """
    Create data element with specified type.
    
    Args:
        _value: Python value to store
        _pytype: Python type name override
        _xsi: XML Schema instance type
        **kwargs: Additional element attributes
    
    Returns:
        ObjectifiedDataElement: Typed data element
    """
```

### Element Creation

Factory classes and functions for creating objectified elements.

```python { .api }
class ElementMaker:
    """Factory for creating objectified elements with namespace support."""
    
    def __init__(self, namespace=None, nsmap=None, makeelement=None, 
                 typemap=None, **kwargs):
        """
        Create element factory.
        
        Args:
            namespace: Default namespace URI
            nsmap: Namespace prefix mapping
            makeelement: Custom element factory function
            typemap: Type mapping for value conversion
            **kwargs: Default attributes for created elements
        """
    
    def __call__(self, tag, *children, **kwargs):
        """Create element with tag, children, and attributes."""
    
    def __getattr__(self, tag):
        """Create element factory function for specific tag."""

# Default element maker instance
E = ElementMaker()

class ObjectPath:
    """
    Object path for navigating objectified XML trees.
    """
    
    def __init__(self, path):
        """
        Create object path from dot-separated string.
        
        Args:
            path: Dot-separated path like "root.child.grandchild"
        """
    
    def find(self, root):
        """Find element at path in objectified tree."""
    
    def setattr(self, root, value):
        """Set value at path, creating elements as needed."""
    
    def hasattr(self, root):
        """Test if path exists in tree."""
```

### Type Annotation

Functions for managing Python type information in XML.

```python { .api }
def annotate(element_or_tree, tag=None, empty_pytype=None, 
             ignore_old=False, ignore_xsi=False, empty_type=None):
    """
    Add Python type annotations to elements based on content.
    
    Args:
        element_or_tree: Element or tree to annotate
        tag: Only annotate elements with this tag
        empty_pytype: Type to use for empty elements
        ignore_old: Ignore existing pytype annotations
        ignore_xsi: Ignore existing xsi:type annotations
        empty_type: Type class for empty elements
    """

def deannotate(element_or_tree, pytype=True, xsi=True, xsi_nil=True, 
               cleanup_namespaces=False):
    """
    Remove type annotations from elements.
    
    Args:
        element_or_tree: Element or tree to process
        pytype: Remove pytype attributes
        xsi: Remove xsi:type attributes  
        xsi_nil: Remove xsi:nil attributes
        cleanup_namespaces: Remove unused namespace declarations
    """

def pyannotate(element_or_tree, ignore_old=False, ignore_xsi=False, 
               empty_type=None):
    """Add Python-specific type annotations."""

def xsiannotate(element_or_tree):
    """Add XML Schema instance type annotations."""
```

### Parser Configuration

Specialized parsers for objectify functionality.

```python { .api }
class ObjectifyElementClassLookup:
    """Element class lookup that creates objectified elements."""
    
    def __init__(self, tree_class=None, empty_data_class=None):
        """
        Create objectify element class lookup.
        
        Args:
            tree_class: Class for tree/document elements
            empty_data_class: Class for empty data elements
        """

def makeparser(**kwargs):
    """
    Create parser configured for objectify processing.
    
    Args:
        **kwargs: Parser configuration options
    
    Returns:
        XMLParser: Parser with ObjectifyElementClassLookup
    """

def set_default_parser(parser):
    """Set default parser for objectify module."""
```

### Utility Functions

Helper functions for working with objectified trees.

```python { .api }
def dump(element_or_tree):
    """Print debug representation of objectified tree."""

def enable_recursive_str(enabled=True):
    """
    Enable/disable recursive string representation.
    
    Args:
        enabled: Enable recursive str() for nested elements
    """

def set_pytype_attribute_tag(attribute_tag=None):
    """
    Configure attribute name for Python type information.
    
    Args:
        attribute_tag: Attribute name (None for default)
    """

def pytypename(obj):
    """
    Get Python type name for object.
    
    Args:
        obj: Python object
    
    Returns:
        str: Type name string
    """

def getRegisteredTypes():
    """
    Get list of registered Python types.
    
    Returns:
        list: Registered type classes
    """
```

### Type Management

Classes for managing Python type information.

```python { .api }
class PyType:
    """Python type annotation handler."""
    
    def __init__(self, name, type_check, type_class, stringify=None):
        """
        Register Python type.
        
        Args:
            name: Type name string
            type_check: Function to test if value matches type
            type_class: Element class for this type
            stringify: Function to convert value to string
        """
    
    @property
    def name(self) -> str:
        """Type name."""
    
    @property  
    def xmlSchemaTypes(self) -> list:
        """Associated XML Schema types."""
```

## Usage Examples

### Basic Object-Oriented Access

```python
from lxml import objectify

# Parse XML with automatic type conversion
xml_data = '''<?xml version="1.0"?>
<catalog>
    <book id="1">
        <title>Python Programming</title>
        <author>John Smith</author>
        <year>2023</year>
        <price>29.99</price>
        <available>true</available>
        <chapters>12</chapters>
    </book>
    <book id="2">
        <title>Web Development</title>
        <author>Jane Doe</author>
        <year>2022</year>
        <price>34.95</price>
        <available>false</available>
        <chapters>15</chapters>
    </book>
</catalog>'''

root = objectify.fromstring(xml_data)

# Access as Python attributes with automatic type conversion
print(root.book[0].title)      # "Python Programming" (string)
print(root.book[0].year)       # 2023 (integer)
print(root.book[0].price)      # 29.99 (float)
print(root.book[0].available)  # True (boolean)
print(root.book[0].chapters)   # 12 (integer)

# Access XML attributes
print(root.book[0].get('id'))  # "1"

# Iterate over multiple elements
for book in root.book:
    print(f"{book.title} ({book.year}): ${book.price}")

# Modify content
root.book[0].price = 24.99
root.book[0].sale = True  # Creates new element

print(objectify.dump(root))
```

### Creating Objectified XML

```python  
from lxml import objectify

# Create root element
root = objectify.Element("products")

# Add child elements with data
root.product = objectify.Element("product", id="1")
root.product.name = "Widget"
root.product.price = 19.99
root.product.in_stock = True
root.product.categories = objectify.Element("categories")
root.product.categories.category = ["Electronics", "Gadgets"]

# Add another product
product2 = objectify.SubElement(root, "product", id="2")
product2.name = "Gadget"
product2.price = 15.50
product2.in_stock = False

# Convert to string
xml_string = objectify.dump(root)
print(xml_string)
```

### Using ElementMaker

```python
from lxml import objectify

# Create custom ElementMaker
E = objectify.ElementMaker(annotate=False)

# Build XML structure
doc = E.order(
    E.id(12345),
    E.customer(
        E.name("John Doe"),
        E.email("john@example.com")
    ),
    E.items(
        E.item(
            E.product("Widget"),
            E.quantity(2),
            E.price(19.99)
        ),
        E.item(
            E.product("Gadget"), 
            E.quantity(1),
            E.price(15.50)
        )
    ),
    E.total(55.48),
    E.date("2023-12-07")
)

# Access created structure
print(f"Order ID: {doc.id}")
print(f"Customer: {doc.customer.name}")
print(f"Total: ${doc.total}")

for item in doc.items.item:
    print(f"- {item.product}: {item.quantity} @ ${item.price}")
```

### Object Path Navigation

```python
from lxml import objectify

xml_data = '''
<config>
    <database>
        <host>localhost</host>
        <port>5432</port>
        <credentials>
            <username>admin</username>
            <password>secret</password>
        </credentials>
    </database>
</config>'''

root = objectify.fromstring(xml_data)

# Create object paths
host_path = objectify.ObjectPath("database.host")
creds_path = objectify.ObjectPath("database.credentials")

# Navigate using paths
print(host_path.find(root))  # "localhost"
creds = creds_path.find(root)
print(f"User: {creds.username}, Pass: {creds.password}")

# Set values using paths
host_path.setattr(root, "prod-server.com")
print(host_path.find(root))  # "prod-server.com"
```

### Type Annotation Control

```python
from lxml import objectify

xml_data = '''
<data>
    <number>42</number>
    <decimal>3.14</decimal>
    <flag>true</flag>
    <text>hello</text>
</data>'''

root = objectify.fromstring(xml_data)

# Check current annotations
print("Before annotation:")
print(objectify.dump(root))

# Add type annotations
objectify.annotate(root)
print("\nAfter annotation:")  
print(objectify.dump(root))

# Remove annotations
objectify.deannotate(root)
print("\nAfter deannotation:")
print(objectify.dump(root))

# Custom type handling
objectify.set_pytype_attribute_tag("data-type")
objectify.annotate(root)
print("\nWith custom type attribute:")
print(objectify.dump(root))
```