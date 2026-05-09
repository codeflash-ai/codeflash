# Schema Validation

Comprehensive XML document validation using multiple schema languages including DTD, RelaxNG, W3C XML Schema, and ISO Schematron. The validation framework provides detailed error reporting, custom validation rules, and integration with parsing workflows.

## Capabilities

### DTD Validation

Document Type Definition validation for XML documents with entity and attribute declarations.

```python { .api }
class DTD:
    """Document Type Definition validator."""
    
    def __init__(self, file=None, external_id=None):
        """
        Create DTD validator.
        
        Args:
            file: Path to DTD file or file-like object
            external_id: External DTD identifier (PUBLIC/SYSTEM)
        """
    
    def validate(self, etree):
        """
        Validate document against DTD.
        
        Args:
            etree: Element or ElementTree to validate
        
        Returns:
            bool: True if valid, False if invalid
        """
    
    @property
    def error_log(self):
        """Validation error log."""
    
    def assertValid(self, etree):
        """Assert document is valid, raise DTDValidateError if not."""

# DTD parsing from strings
def DTD(file=None, external_id=None):
    """Create DTD validator from file or external identifier."""
```

### RelaxNG Validation

RELAX NG schema validation with compact and XML syntax support.

```python { .api }
class RelaxNG:
    """RELAX NG schema validator."""
    
    def __init__(self, etree=None, file=None):
        """
        Create RelaxNG validator.
        
        Args:
            etree: Element or ElementTree containing schema
            file: Path to schema file or file-like object
        """
    
    def validate(self, etree):
        """
        Validate document against RelaxNG schema.
        
        Args:
            etree: Element or ElementTree to validate
        
        Returns:
            bool: True if valid, False if invalid
        """
    
    @property
    def error_log(self):
        """Validation error log."""
    
    def assertValid(self, etree):
        """Assert document is valid, raise RelaxNGValidateError if not."""

# Factory function
def RelaxNG(etree=None, file=None):
    """Create RelaxNG validator from schema document or file."""
```

### XML Schema Validation

W3C XML Schema validation with full XSD 1.0 support.

```python { .api }
class XMLSchema:
    """W3C XML Schema validator."""
    
    def __init__(self, etree=None, file=None):
        """
        Create XMLSchema validator.
        
        Args:
            etree: Element or ElementTree containing schema
            file: Path to schema file or file-like object
        """
    
    def validate(self, etree):
        """
        Validate document against XML Schema.
        
        Args:
            etree: Element or ElementTree to validate
        
        Returns:
            bool: True if valid, False if invalid
        """
    
    @property
    def error_log(self):
        """Validation error log."""
    
    def assertValid(self, etree):
        """Assert document is valid, raise XMLSchemaValidateError if not."""

# Factory function  
def XMLSchema(etree=None, file=None):
    """Create XMLSchema validator from schema document or file."""
```

### Schematron Validation

ISO Schematron rule-based validation with XPath assertions.

```python { .api }
class Schematron:
    """ISO Schematron validator."""
    
    def __init__(self, etree=None, file=None, include=True, expand=True,
                 include_params=None, expand_params=None, compile_params=None,
                 store_schematron=False, store_xslt=False, store_report=False,
                 phase=None, error_finder=None):
        """
        Create Schematron validator.
        
        Args:
            etree: Element or ElementTree containing schema
            file: Path to schema file or file-like object
            include: Process schematron includes (step 1)
            expand: Expand abstract patterns (step 2)
            include_params: Parameters for include step
            expand_params: Parameters for expand step
            compile_params: Parameters for compile step
            store_schematron: Keep processed schematron document
            store_xslt: Keep compiled XSLT stylesheet
            store_report: Keep validation report
            phase: Schematron validation phase
            error_finder: Custom error finder XPath
        """
    
    def validate(self, etree):
        """
        Validate document against Schematron rules.
        
        Args:
            etree: Element or ElementTree to validate
        
        Returns:
            bool: True if valid, False if invalid
        """
    
    @property
    def error_log(self):
        """Validation error log."""
    
    @property
    def schematron(self):
        """Processed schematron document (if stored)."""
    
    @property
    def validator_xslt(self):
        """Compiled XSLT validator (if stored)."""
    
    @property
    def validation_report(self):
        """SVRL validation report (if stored)."""
    
    def assertValid(self, etree):
        """Assert document is valid, raise SchematronValidateError if not."""
    
    # Class constants for error handling
    ASSERTS_ONLY = None      # Report failed assertions only (default)
    ASSERTS_AND_REPORTS = None  # Report assertions and successful reports

# Schematron processing functions
def extract_xsd(schema_doc):
    """Extract embedded schematron from XML Schema."""

def extract_rng(schema_doc):  
    """Extract embedded schematron from RelaxNG schema."""

def iso_dsdl_include(schematron_doc, **params):
    """Process schematron include directives."""

def iso_abstract_expand(schematron_doc, **params):
    """Expand abstract patterns in schematron."""

def iso_svrl_for_xslt1(schematron_doc, **params):
    """Compile schematron to XSLT validation stylesheet."""

def stylesheet_params(**kwargs):
    """Convert keyword arguments to XSLT stylesheet parameters."""
```

### Validation Error Handling

Comprehensive error classes for different validation failures.

```python { .api }
class DocumentInvalid(LxmlError):
    """Base class for document validation errors."""

class DTDError(LxmlError):
    """Base class for DTD-related errors."""

class DTDParseError(DTDError):
    """DTD parsing error."""

class DTDValidateError(DTDError, DocumentInvalid):
    """DTD validation error."""

class RelaxNGError(LxmlError):
    """Base class for RelaxNG-related errors."""

class RelaxNGParseError(RelaxNGError):
    """RelaxNG schema parsing error."""

class RelaxNGValidateError(RelaxNGError, DocumentInvalid):
    """RelaxNG validation error."""

class XMLSchemaError(LxmlError):
    """Base class for XML Schema-related errors."""

class XMLSchemaParseError(XMLSchemaError):
    """XML Schema parsing error."""

class XMLSchemaValidateError(XMLSchemaError, DocumentInvalid):
    """XML Schema validation error."""

class SchematronError(LxmlError):
    """Base class for Schematron-related errors."""

class SchematronParseError(SchematronError):
    """Schematron schema parsing error."""

class SchematronValidateError(SchematronError, DocumentInvalid):
    """Schematron validation error."""
```

### Parser Integration

Integrate validation directly into parsing workflow.

```python { .api }
class XMLParser:
    """XML parser with validation support."""
    
    def __init__(self, dtd_validation=False, schema=None, **kwargs):
        """
        Create parser with validation options.
        
        Args:
            dtd_validation: Enable DTD validation during parsing
            schema: Validator instance (RelaxNG, XMLSchema, etc.)
            **kwargs: Other parser options
        """

# Validation during parsing
def parse(source, parser=None, base_url=None):
    """Parse with validation if parser configured."""

def fromstring(text, parser=None, base_url=None):
    """Parse string with validation if parser configured."""
```

## Usage Examples

### DTD Validation

```python
from lxml import etree

# DTD schema
dtd_content = '''
<!ELEMENT catalog (book+)>
<!ELEMENT book (title, author, year, price)>
<!ATTLIST book id CDATA #REQUIRED
               category (fiction|science|mystery) #REQUIRED>
<!ELEMENT title (#PCDATA)>
<!ELEMENT author (#PCDATA)>
<!ELEMENT year (#PCDATA)>
<!ELEMENT price (#PCDATA)>
<!ATTLIST price currency CDATA #IMPLIED>
'''

# XML document
xml_content = '''<?xml version="1.0"?>
<!DOCTYPE catalog [
''' + dtd_content + '''
]>
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
        <price>15.99</price>
    </book>
</catalog>'''

# Parse and validate
parser = etree.XMLParser(dtd_validation=True)
try:
    doc = etree.fromstring(xml_content, parser)
    print("Document is valid according to DTD")
except etree.DTDValidateError as e:
    print(f"DTD validation failed: {e}")

# Separate DTD validation
dtd = etree.DTD(external_id=None)  # Would load from DOCTYPE
doc = etree.fromstring(xml_content)
if dtd.validate(doc):
    print("Document is valid")
else:
    print("Validation errors:")
    for error in dtd.error_log:
        print(f"  Line {error.line}: {error.message}")
```

### RelaxNG Validation

```python
from lxml import etree

# RelaxNG schema
relaxng_schema = '''
<element name="catalog" xmlns="http://relaxng.org/ns/structure/1.0">
    <oneOrMore>
        <element name="book">
            <attribute name="id"/>
            <attribute name="category">
                <choice>
                    <value>fiction</value>
                    <value>science</value>
                    <value>mystery</value>
                </choice>
            </attribute>
            <element name="title"><text/></element>
            <element name="author"><text/></element>
            <element name="year"><text/></element>
            <element name="price">
                <optional>
                    <attribute name="currency"/>
                </optional>
                <text/>
            </element>
        </element>
    </oneOrMore>
</element>
'''

# Create validator
relaxng_doc = etree.fromstring(relaxng_schema)
relaxng = etree.RelaxNG(relaxng_doc)

# XML to validate
xml_content = '''
<catalog>
    <book id="1" category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <price currency="USD">12.99</price>
    </book>
</catalog>
'''

# Validate
doc = etree.fromstring(xml_content)
if relaxng.validate(doc):
    print("Document is valid according to RelaxNG")
else:
    print("RelaxNG validation errors:")
    for error in relaxng.error_log:
        print(f"  Line {error.line}: {error.message}")

# Use with parser
parser = etree.XMLParser(schema=relaxng)
try:
    validated_doc = etree.fromstring(xml_content, parser)
    print("Document parsed and validated successfully")
except etree.RelaxNGValidateError as e:
    print(f"Validation during parsing failed: {e}")
```

### XML Schema Validation

```python
from lxml import etree

# XML Schema (XSD)
xsd_schema = '''<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xs:element name="catalog">
        <xs:complexType>
            <xs:sequence>
                <xs:element name="book" maxOccurs="unbounded">
                    <xs:complexType>
                        <xs:sequence>
                            <xs:element name="title" type="xs:string"/>
                            <xs:element name="author" type="xs:string"/>
                            <xs:element name="year" type="xs:gYear"/>
                            <xs:element name="price">
                                <xs:complexType>
                                    <xs:simpleContent>
                                        <xs:extension base="xs:decimal">
                                            <xs:attribute name="currency" type="xs:string"/>
                                        </xs:extension>
                                    </xs:simpleContent>
                                </xs:complexType>
                            </xs:element>
                        </xs:sequence>
                        <xs:attribute name="id" type="xs:string" use="required"/>
                        <xs:attribute name="category" use="required">
                            <xs:simpleType>
                                <xs:restriction base="xs:string">
                                    <xs:enumeration value="fiction"/>
                                    <xs:enumeration value="science"/>
                                    <xs:enumeration value="mystery"/>
                                </xs:restriction>
                            </xs:simpleType>
                        </xs:attribute>
                    </xs:complexType>
                </xs:element>
            </xs:sequence>
        </xs:complexType>
    </xs:element>
</xs:schema>
'''

# Create XML Schema validator
xsd_doc = etree.fromstring(xsd_schema)
xmlschema = etree.XMLSchema(xsd_doc)

# Validate document
xml_content = '''
<catalog>
    <book id="1" category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <price currency="USD">12.99</price>
    </book>
</catalog>
'''

doc = etree.fromstring(xml_content)
if xmlschema.validate(doc):
    print("Document is valid according to XML Schema")
else:
    print("XML Schema validation errors:")
    for error in xmlschema.error_log:
        print(f"  Line {error.line}: {error.message}")
```

### Schematron Validation

```python
from lxml import etree
from lxml.isoschematron import Schematron

# Schematron schema with business rules
schematron_schema = '''<?xml version="1.0"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron">
    <title>Book Catalog Validation</title>
    
    <pattern id="price-rules">
        <title>Price validation rules</title>
        
        <rule context="book">
            <assert test="price[@currency]">
                Books should have currency specified for price
            </assert>
            <assert test="number(price) > 0">
                Book price must be positive: <value-of select="title"/>
            </assert>
            <assert test="number(price) &lt; 100">
                Book price seems too high: <value-of select="title"/> costs <value-of select="price"/>
            </assert>
        </rule>
        
        <rule context="book[@category='fiction']">
            <assert test="number(year) >= 1800">
                Fiction books should be from 1800 or later
            </assert>
        </rule> 
        
        <rule context="book[@category='science']">
            <assert test="number(year) >= 1900">
                Science books should be relatively recent (1900+)
            </assert>
        </rule>
    </pattern>
</schema>
'''

# Create Schematron validator
schematron_doc = etree.fromstring(schematron_schema)
schematron = Schematron(schematron_doc)

# Test valid document
valid_xml = '''
<catalog>
    <book id="1" category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <year>1925</year>
        <price currency="USD">12.99</price>
    </book>
</catalog>
'''

doc = etree.fromstring(valid_xml)
if schematron.validate(doc):
    print("Document passes Schematron validation")
else:
    print("Schematron validation errors:")
    for error in schematron.error_log:
        print(f"  {error.message}")

# Test invalid document
invalid_xml = '''
<catalog>
    <book id="1" category="science">
        <title>Ancient Science</title>
        <author>Old Author</author>
        <year>1850</year>
        <price>-5.99</price>
    </book>
</catalog>
'''

doc = etree.fromstring(invalid_xml)
if not schematron.validate(doc):
    print("\nSchematron validation failed as expected:")
    for error in schematron.error_log:
        print(f"  {error.message}")
```

### Combined Validation

```python
from lxml import etree
from lxml.isoschematron import Schematron

# Multi-step validation: structure + business rules
def validate_document(xml_content, relaxng_schema, schematron_schema):
    """Validate document against both structural and business rules."""
    
    doc = etree.fromstring(xml_content)
    
    # Step 1: Structural validation with RelaxNG
    relaxng = etree.RelaxNG(etree.fromstring(relaxng_schema))
    if not relaxng.validate(doc):
        return False, "Structural validation failed", relaxng.error_log
    
    # Step 2: Business rules validation with Schematron
    schematron = Schematron(etree.fromstring(schematron_schema))
    if not schematron.validate(doc):
        return False, "Business rules validation failed", schematron.error_log
    
    return True, "Document is fully valid", None

# Use combined validation
xml_to_test = '''
<catalog>
    <book id="1" category="fiction">
        <title>Test Book</title>
        <author>Test Author</author>
        <year>2023</year>
        <price currency="USD">25.99</price>
    </book>
</catalog>
'''

is_valid, message, errors = validate_document(
    xml_to_test, relaxng_schema, schematron_schema
)

print(f"Validation result: {message}")
if errors:
    for error in errors:
        print(f"  {error.message}")
```