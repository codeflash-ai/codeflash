# Text Utilities

Functions for working with text positions, line/character coordinates, and character classification.

## Position Utilities

```typescript { .api }
function getPositionOfLineAndCharacter(
  sourceFile: SourceFileLike,
  line: number,
  character: number
): number;

function getLineAndCharacterOfPosition(
  sourceFile: SourceFileLike,
  position: number
): LineAndCharacter;

interface LineAndCharacter {
  line: number;
  character: number;
}
```

## Character Classification

```typescript { .api }
function isWhiteSpaceLike(ch: number): boolean;

function isWhiteSpaceSingleLine(ch: number): boolean;

function isLineBreak(ch: number): boolean;

function isDigit(ch: number): boolean;

function isOctalDigit(ch: number): boolean;

function tokenToString(t: SyntaxKind): string | undefined;
```

## Usage Example

```typescript
import * as ts from 'typescript';

const sourceCode = `function hello() {
  console.log("Hello");
}`;

const sourceFile = ts.createSourceFile(
  'example.ts',
  sourceCode,
  ts.ScriptTarget.Latest,
  true
);

const position = 25; // Position of 'console'
const lineChar = ts.getLineAndCharacterOfPosition(sourceFile, position);
console.log(`Position ${position} is at line ${lineChar.line + 1}, character ${lineChar.character + 1}`);
// Output: Position 25 is at line 2, character 3

const pos = ts.getPositionOfLineAndCharacter(sourceFile, 1, 2); // Line 2, char 3 (0-indexed)
console.log(`Line 2, character 3 is at position ${pos}`);
// Output: Line 2, character 3 is at position 25

console.log(ts.isWhiteSpaceLike(' '.charCodeAt(0)));  // true
console.log(ts.isLineBreak('\n'.charCodeAt(0))); // true
console.log(ts.isDigit('5'.charCodeAt(0))); // true
```

