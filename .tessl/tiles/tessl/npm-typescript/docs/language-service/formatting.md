# Formatting

Code formatting and editor settings APIs for controlling whitespace, indentation, and code style preferences. The formatting system is integrated with the Language Service for IDE features.

## Capabilities

### FormatCodeSettings Interface

Comprehensive formatting configuration with 20+ options for controlling code style.

```typescript { .api }
interface FormatCodeSettings extends EditorSettings {
  readonly insertSpaceAfterCommaDelimiter?: boolean;
  readonly insertSpaceAfterSemicolonInForStatements?: boolean;
  readonly insertSpaceBeforeAndAfterBinaryOperators?: boolean;
  readonly insertSpaceAfterConstructor?: boolean;
  readonly insertSpaceAfterKeywordsInControlFlowStatements?: boolean;
  readonly insertSpaceAfterFunctionKeywordForAnonymousFunctions?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingNonemptyParenthesis?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingNonemptyBrackets?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingNonemptyBraces?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingEmptyBraces?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingTemplateStringBraces?: boolean;
  readonly insertSpaceAfterOpeningAndBeforeClosingJsxExpressionBraces?: boolean;
  readonly insertSpaceAfterTypeAssertion?: boolean;
  readonly insertSpaceBeforeFunctionParenthesis?: boolean;
  readonly placeOpenBraceOnNewLineForFunctions?: boolean;
  readonly placeOpenBraceOnNewLineForControlBlocks?: boolean;
  readonly insertSpaceBeforeTypeAnnotation?: boolean;
  readonly indentMultiLineObjectLiteralBeginningOnBlankLine?: boolean;
  readonly semicolons?: SemicolonPreference;
  readonly indentSwitchCase?: boolean;
}
```

### EditorSettings Interface

Base editor configuration for indentation and line handling.

```typescript { .api }
interface EditorSettings {
  baseIndentSize?: number;
  indentSize?: number;
  tabSize?: number;
  newLineCharacter?: string;
  convertTabsToSpaces?: boolean;
  indentStyle?: IndentStyle;
  trimTrailingWhitespace?: boolean;
}
```

### Formatting Enums

Enums controlling indentation and semicolon behavior.

```typescript { .api }
enum IndentStyle {
  None = 0,
  Block = 1,
  Smart = 2
}

enum SemicolonPreference {
  Ignore = "ignore",
  Insert = "insert",
  Remove = "remove"
}
```

### Language Service Formatting Methods

Formatting operations provided by the Language Service.

```typescript { .api }
interface LanguageService {
  /**
   * Format a range of text in a file.
   */
  getFormattingEditsForRange(
    fileName: string,
    start: number,
    end: number,
    options: FormatCodeSettings
  ): TextChange[];

  /**
   * Format the entire document.
   */
  getFormattingEditsForDocument(
    fileName: string,
    options: FormatCodeSettings
  ): TextChange[];

  /**
   * Format text after a key is typed.
   */
  getFormattingEditsAfterKeystroke(
    fileName: string,
    position: number,
    key: string,
    options: FormatCodeSettings
  ): TextChange[];

  /**
   * Get format options from Language Service host.
   */
  getFormatCodeOptions?(fileName: string): FormatCodeSettings;
}
```

### Text Changes

Structure representing formatting edits.

```typescript { .api }
interface TextChange {
  span: TextSpan;
  newText: string;
}

interface TextSpan {
  start: number;
  length: number;
}
```

## Usage Examples

### Format a Document

Format an entire TypeScript file:

```typescript
import * as ts from 'typescript';

// Create language service
const services = ts.createLanguageService({
  getScriptFileNames: () => ['example.ts'],
  getScriptVersion: () => '1',
  getScriptSnapshot: (fileName) => {
    if (fileName === 'example.ts') {
      return ts.ScriptSnapshot.fromString(
        'function foo( x:number,y:string ){return x+y;}'
      );
    }
    return undefined;
  },
  getCurrentDirectory: () => '',
  getCompilationSettings: () => ({}),
  getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
  fileExists: ts.sys.fileExists,
  readFile: ts.sys.readFile,
  readDirectory: ts.sys.readDirectory
});

// Define formatting options
const formatSettings: ts.FormatCodeSettings = {
  indentSize: 2,
  tabSize: 2,
  convertTabsToSpaces: true,
  insertSpaceAfterCommaDelimiter: true,
  insertSpaceAfterSemicolonInForStatements: true,
  insertSpaceBeforeAndAfterBinaryOperators: true,
  insertSpaceAfterKeywordsInControlFlowStatements: true,
  insertSpaceAfterFunctionKeywordForAnonymousFunctions: false,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyParenthesis: false,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyBrackets: false,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyBraces: true,
  insertSpaceBeforeTypeAnnotation: true,
  placeOpenBraceOnNewLineForFunctions: false,
  placeOpenBraceOnNewLineForControlBlocks: false,
  semicolons: ts.SemicolonPreference.Insert
};

// Get formatting edits for the entire document
const edits = services.getFormattingEditsForDocument(
  'example.ts',
  formatSettings
);

// Apply the edits
let formattedCode = 'function foo( x:number,y:string ){return x+y;}';
for (let i = edits.length - 1; i >= 0; i--) {
  const edit = edits[i];
  formattedCode =
    formattedCode.substring(0, edit.span.start) +
    edit.newText +
    formattedCode.substring(edit.span.start + edit.span.length);
}

console.log(formattedCode);
// Output: function foo(x: number, y: string) { return x + y; }
```

### Format a Range

Format only a specific range of code:

```typescript
import * as ts from 'typescript';

// Assume language service is set up as above

const source = `function foo() {
const x=1;
const y    =    2;
return x+y;
}`;

// Format only lines containing the variable declarations (positions 20-60)
const rangeEdits = services.getFormattingEditsForRange(
  'example.ts',
  20,
  60,
  {
    indentSize: 2,
    insertSpaceBeforeAndAfterBinaryOperators: true
  }
);

// Apply edits to get formatted range
// The function body will be formatted, but not the function declaration
```

### Format After Keystroke

Automatically format code when user types specific characters:

```typescript
import * as ts from 'typescript';

function formatOnKeystroke(
  services: ts.LanguageService,
  fileName: string,
  position: number,
  key: string,
  formatSettings: ts.FormatCodeSettings
): string {
  // Get formatting edits triggered by keystroke
  const edits = services.getFormattingEditsAfterKeystroke(
    fileName,
    position,
    key,
    formatSettings
  );

  if (edits.length === 0) {
    return 'No formatting needed';
  }

  // Apply the edits
  const snapshot = services.getProgram()!.getSourceFile(fileName)!.text;
  let formatted = snapshot;

  for (let i = edits.length - 1; i >= 0; i--) {
    const edit = edits[i];
    formatted =
      formatted.substring(0, edit.span.start) +
      edit.newText +
      formatted.substring(edit.span.start + edit.span.length);
  }

  return formatted;
}

// Example: Format when user types semicolon
const result = formatOnKeystroke(
  services,
  'example.ts',
  25, // position after semicolon
  ';',
  formatSettings
);
```

### Custom Formatting Configuration

Create different formatting profiles for different code styles:

```typescript
import * as ts from 'typescript';

// Compact style
const compactStyle: ts.FormatCodeSettings = {
  indentSize: 2,
  insertSpaceAfterCommaDelimiter: true,
  insertSpaceBeforeAndAfterBinaryOperators: true,
  insertSpaceAfterKeywordsInControlFlowStatements: true,
  insertSpaceAfterFunctionKeywordForAnonymousFunctions: false,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyParenthesis: false,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyBraces: false,
  placeOpenBraceOnNewLineForFunctions: false,
  placeOpenBraceOnNewLineForControlBlocks: false,
  semicolons: ts.SemicolonPreference.Remove
};

// Spacious style
const spaciousStyle: ts.FormatCodeSettings = {
  indentSize: 4,
  insertSpaceAfterCommaDelimiter: true,
  insertSpaceAfterSemicolonInForStatements: true,
  insertSpaceBeforeAndAfterBinaryOperators: true,
  insertSpaceAfterKeywordsInControlFlowStatements: true,
  insertSpaceAfterFunctionKeywordForAnonymousFunctions: true,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyParenthesis: true,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyBrackets: true,
  insertSpaceAfterOpeningAndBeforeClosingNonemptyBraces: true,
  placeOpenBraceOnNewLineForFunctions: true,
  placeOpenBraceOnNewLineForControlBlocks: true,
  semicolons: ts.SemicolonPreference.Insert,
  indentSwitchCase: true
};

// Allman brace style
const allmanStyle: ts.FormatCodeSettings = {
  indentSize: 4,
  placeOpenBraceOnNewLineForFunctions: true,
  placeOpenBraceOnNewLineForControlBlocks: true,
  insertSpaceBeforeAndAfterBinaryOperators: true,
  semicolons: ts.SemicolonPreference.Insert
};
```

### Integration with tsconfig.json

Load formatting settings from editor configuration:

```typescript
import * as ts from 'typescript';

function getFormatSettingsFromConfig(
  configPath: string
): ts.FormatCodeSettings {
  // Parse tsconfig.json
  const configFile = ts.readConfigFile(configPath, ts.sys.readFile);

  if (configFile.error) {
    throw new Error(`Failed to read config: ${configFile.error.messageText}`);
  }

  // Extract editor settings
  const editorSettings = configFile.config.formatOptions || {};

  return {
    indentSize: editorSettings.indentSize || 4,
    tabSize: editorSettings.tabSize || 4,
    convertTabsToSpaces: editorSettings.convertTabsToSpaces !== false,
    insertSpaceAfterCommaDelimiter:
      editorSettings.insertSpaceAfterCommaDelimiter !== false,
    insertSpaceBeforeAndAfterBinaryOperators:
      editorSettings.insertSpaceBeforeAndAfterBinaryOperators !== false,
    semicolons: editorSettings.semicolons || ts.SemicolonPreference.Ignore
  };
}
```
