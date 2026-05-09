# Scanner and Parser

Low-level lexical analysis and parsing APIs for tokenizing and parsing TypeScript source code.

## Capabilities

### Scanner Creation

Create a scanner for lexical analysis.

```typescript { .api }
function createScanner(
  languageVersion: ScriptTarget,
  skipTrivia: boolean,
  languageVariant?: LanguageVariant,
  textInitial?: string,
  onError?: ErrorCallback,
  start?: number,
  length?: number
): Scanner;

type ErrorCallback = (message: DiagnosticMessage, length: number, arg0?: any) => void;
```

### Scanner Interface

```typescript { .api }
interface Scanner {
  /** @deprecated use getTokenFullStart */
  getStartPos(): number;
  getToken(): SyntaxKind;
  getTokenFullStart(): number;
  getTokenStart(): number;
  getTokenEnd(): number;
  /** @deprecated use getTokenEnd */
  getTextPos(): number;
  /** @deprecated use getTokenStart */
  getTokenPos(): number;
  getTokenText(): string;
  getTokenValue(): string;
  hasUnicodeEscape(): boolean;
  hasExtendedUnicodeEscape(): boolean;
  hasPrecedingLineBreak(): boolean;
  isIdentifier(): boolean;
  isReservedWord(): boolean;
  isUnterminated(): boolean;
  reScanGreaterToken(): SyntaxKind;
  reScanSlashToken(): SyntaxKind;
  reScanAsteriskEqualsToken(): SyntaxKind;
  reScanTemplateToken(isTaggedTemplate: boolean): SyntaxKind;
  /** @deprecated use reScanTemplateToken(false) */
  reScanTemplateHeadOrNoSubstitutionTemplate(): SyntaxKind;
  scanJsxIdentifier(): SyntaxKind;
  scanJsxAttributeValue(): SyntaxKind;
  reScanJsxAttributeValue(): SyntaxKind;
  reScanJsxToken(allowMultilineJsxText?: boolean): JsxTokenSyntaxKind;
  reScanLessThanToken(): SyntaxKind;
  reScanHashToken(): SyntaxKind;
  reScanQuestionToken(): SyntaxKind;
  reScanInvalidIdentifier(): SyntaxKind;
  scanJsxToken(): JsxTokenSyntaxKind;
  scanJsDocToken(): JSDocSyntaxKind;
  scan(): SyntaxKind;
  getText(): string;
  setText(text: string | undefined, start?: number, length?: number): void;
  setOnError(onError: ErrorCallback | undefined): void;
  setScriptTarget(scriptTarget: ScriptTarget): void;
  setLanguageVariant(variant: LanguageVariant): void;
  setScriptKind(scriptKind: ScriptKind): void;
  setJSDocParsingMode(kind: JSDocParsingMode): void;
  /** @deprecated use resetTokenState */
  setTextPos(textPos: number): void;
  resetTokenState(pos: number): void;
  lookAhead<T>(callback: () => T): T;
  scanRange<T>(start: number, length: number, callback: () => T): T;
  tryScan<T>(callback: () => T): T;
}
```

### Source File Creation

Parse source code into an AST.

```typescript { .api }
function createSourceFile(
  fileName: string,
  sourceText: string,
  languageVersionOrOptions: ScriptTarget | CreateSourceFileOptions,
  setParentNodes?: boolean,
  scriptKind?: ScriptKind
): SourceFile;

interface CreateSourceFileOptions {
  languageVersion: ScriptTarget;
  impliedNodeFormat?: ResolutionMode;
  setExternalModuleIndicator?: (file: SourceFile) => void;
  jsDocParsingMode?: JSDocParsingMode;
}
```

### Source File Interface

```typescript { .api }
interface SourceFile extends Declaration {
  readonly kind: SyntaxKind.SourceFile;
  readonly statements: NodeArray<Statement>;
  readonly endOfFileToken: Token<SyntaxKind.EndOfFileToken>;
  readonly fileName: string;
  readonly text: string;
  readonly amdDependencies: readonly AmdDependency[];
  readonly moduleName?: string;
  readonly referencedFiles: readonly FileReference[];
  readonly typeReferenceDirectives: readonly FileReference[];
  readonly libReferenceDirectives: readonly FileReference[];
  readonly languageVariant: LanguageVariant;
  readonly isDeclarationFile: boolean;
  readonly hasNoDefaultLib: boolean;
  readonly languageVersion: ScriptTarget;
  readonly scriptKind: ScriptKind;
  readonly externalModuleIndicator?: Node;
  readonly commonJsModuleIndicator?: Node;
  readonly identifiers: Map<string, string>;
  readonly nodeCount: number;
  readonly identifierCount: number;
  readonly symbolCount: number;
  readonly parseDiagnostics: DiagnosticWithLocation[];
  readonly bindDiagnostics: readonly Diagnostic[];
  readonly bindSuggestionDiagnostics?: readonly Diagnostic[];
  readonly jsDocDiagnostics?: readonly Diagnostic[];
  readonly additionalSyntacticDiagnostics?: readonly DiagnosticWithLocation[];
  readonly lineMap: readonly number[];
  
  getLineAndCharacterOfPosition(pos: number): LineAndCharacter;
  getLineEndOfPosition(pos: number): number;
  getLineStarts(): readonly number[];
  getPositionOfLineAndCharacter(line: number, character: number): number;
  update(newText: string, textChangeRange: TextChangeRange): SourceFile;
}

interface SourceFileLike {
  readonly text: string;
  readonly lineMap?: readonly number[];
  getLineAndCharacterOfPosition(pos: number): LineAndCharacter;
}
```

### Language Variant

```typescript { .api }
enum LanguageVariant {
  Standard = 0,
  JSX = 1
}

enum ScriptKind {
  Unknown = 0,
  JS = 1,
  JSX = 2,
  TS = 3,
  TSX = 4,
  External = 5,
  JSON = 6,
  Deferred = 7
}
```

### JSDoc Parsing

```typescript { .api }
enum JSDocParsingMode {
  ParseAll = 0,
  ParseNone = 1,
  ParseForTypeErrors = 2,
  ParseForTypeInfo = 3
}

function parseIsolatedEntityName(text: string, languageVersion: ScriptTarget): EntityName | undefined;
function parseJsonText(fileName: string, sourceText: string): JsonSourceFile;
```

### Pre-Processed File Info

Get file dependencies without full parsing.

```typescript { .api }
function preProcessFile(
  sourceText: string,
  readImportFiles?: boolean,
  detectJavaScriptImports?: boolean
): PreProcessedFileInfo;

interface PreProcessedFileInfo {
  referencedFiles: FileReference[];
  typeReferenceDirectives: FileReference[];
  libReferenceDirectives: FileReference[];
  importedFiles: FileReference[];
  ambientExternalModules?: string[];
  isLibFile: boolean;
}

interface FileReference extends TextRange {
  fileName: string;
  resolutionMode?: ResolutionMode;
}
```

### Text Range

```typescript { .api }
interface TextRange {
  pos: number;
  end: number;
}

interface ReadonlyTextRange {
  readonly pos: number;
  readonly end: number;
}

interface TextSpan {
  start: number;
  length: number;
}

interface TextChangeRange {
  span: TextSpan;
  newLength: number;
}

function createTextSpan(start: number, length: number): TextSpan;
function createTextSpanFromBounds(start: number, end: number): TextSpan;
function createTextChangeRange(span: TextSpan, newLength: number): TextChangeRange;
function collapseTextChangeRangesAcrossMultipleVersions(changes: readonly TextChangeRange[]): TextChangeRange;
```

### Comment Ranges

```typescript { .api }
function getLeadingCommentRanges(text: string, pos: number): CommentRange[] | undefined;
function getTrailingCommentRanges(text: string, pos: number): CommentRange[] | undefined;
function getShebang(text: string): string | undefined;
function isIdentifierStart(ch: number, languageVersion: ScriptTarget | undefined): boolean;
function isIdentifierPart(ch: number, languageVersion: ScriptTarget | undefined, identifierVariant?: LanguageVariant): boolean;

interface CommentRange extends TextRange {
  kind: CommentKind;
  hasTrailingNewLine?: boolean;
}

enum CommentKind {
  SingleLine = 2,
  MultiLine = 3
}
```

### Token Utilities

```typescript { .api }
function isKeyword(kind: SyntaxKind): boolean;
function isModifier(kind: SyntaxKind): boolean;
function isFutureReservedKeyword(kind: SyntaxKind): boolean;
function isContextualKeyword(kind: SyntaxKind): boolean;
function isStringANonContextualKeyword(name: string): boolean;
function isStringAKeyword(name: string): boolean;
function isIdentifierANonContextualKeyword(node: Identifier): boolean;
function isTrivia(kind: SyntaxKind): boolean;
function isPunctuation(kind: SyntaxKind): boolean;
function couldStartTrivia(text: string, pos: number): boolean;
function scanString(text: string, start: number, astral?: boolean): string | undefined;
function getTokenText(kind: SyntaxKind): string | undefined;
```

## Usage Example

Using the scanner to tokenize source code:

```typescript
import * as ts from 'typescript';

const code = 'const x: number = 42;';
const scanner = ts.createScanner(
  ts.ScriptTarget.Latest,
  false, // don't skip trivia
  ts.LanguageVariant.Standard,
  code
);

let token: ts.SyntaxKind;
while ((token = scanner.scan()) !== ts.SyntaxKind.EndOfFileToken) {
  const start = scanner.getTokenStart();
  const end = scanner.getTokenEnd();
  const text = scanner.getTokenText();
  
  console.log(
    ts.SyntaxKind[token],
    `"${text}"`,
    `(${start}-${end})`
  );
}
```

Parsing source code:

```typescript
import * as ts from 'typescript';

const code = `
function greet(name: string): string {
  return 'Hello, ' + name;
}
`;

const sourceFile = ts.createSourceFile(
  'example.ts',
  code,
  ts.ScriptTarget.Latest,
  true // set parent nodes
);

console.log('File name:', sourceFile.fileName);
console.log('Statements:', sourceFile.statements.length);
console.log('Parse diagnostics:', sourceFile.parseDiagnostics.length);

// Print line and character for position 0
const lineChar = sourceFile.getLineAndCharacterOfPosition(0);
console.log(`Position 0 is at line ${lineChar.line}, character ${lineChar.character}`);
```
