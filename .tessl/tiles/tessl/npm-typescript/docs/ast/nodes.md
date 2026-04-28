# AST Nodes

TypeScript's Abstract Syntax Tree (AST) node types and traversal APIs with 359 syntax kinds.

## Capabilities

### SyntaxKind Enum

Complete enumeration of all syntax node types.

```typescript { .api }
enum SyntaxKind {
  Unknown = 0,
  EndOfFileToken = 1,
  SingleLineCommentTrivia = 2,
  MultiLineCommentTrivia = 3,
  NewLineTrivia = 4,
  WhitespaceTrivia = 5,
  ShebangTrivia = 6,
  ConflictMarkerTrivia = 7,
  NonTextFileMarkerTrivia = 8,
  NumericLiteral = 9,
  BigIntLiteral = 10,
  StringLiteral = 11,
  JsxText = 12,
  JsxTextAllWhiteSpaces = 13,
  RegularExpressionLiteral = 14,
  NoSubstitutionTemplateLiteral = 15,
  TemplateHead = 16,
  TemplateMiddle = 17,
  TemplateTail = 18,
  OpenBraceToken = 19,
  CloseBraceToken = 20,
  OpenParenToken = 21,
  CloseParenToken = 22,
  OpenBracketToken = 23,
  CloseBracketToken = 24,
  DotToken = 25,
  DotDotDotToken = 26,
  SemicolonToken = 27,
  CommaToken = 28,
  QuestionDotToken = 29,
  LessThanToken = 30,
  LessThanSlashToken = 31,
  GreaterThanToken = 32,
  LessThanEqualsToken = 33,
  GreaterThanEqualsToken = 34,
  EqualsEqualsToken = 35,
  ExclamationEqualsToken = 36,
  EqualsEqualsEqualsToken = 37,
  ExclamationEqualsEqualsToken = 38,
  EqualsGreaterThanToken = 39,
  PlusToken = 40,
  MinusToken = 41,
  AsteriskToken = 42,
  AsteriskAsteriskToken = 43,
  SlashToken = 44,
  PercentToken = 45,
  PlusPlusToken = 46,
  MinusMinusToken = 47,
  LessThanLessThanToken = 48,
  GreaterThanGreaterThanToken = 49,
  GreaterThanGreaterThanGreaterThanToken = 50,
  AmpersandToken = 51,
  BarToken = 52,
  CaretToken = 53,
  ExclamationToken = 54,
  TildeToken = 55,
  AmpersandAmpersandToken = 56,
  BarBarToken = 57,
  QuestionToken = 58,
  ColonToken = 59,
  AtToken = 60,
  QuestionQuestionToken = 61,
  BacktickToken = 62,
  HashToken = 63,
  EqualsToken = 64,
  PlusEqualsToken = 65,
  MinusEqualsToken = 66,
  AsteriskEqualsToken = 67,
  AsteriskAsteriskEqualsToken = 68,
  SlashEqualsToken = 69,
  PercentEqualsToken = 70,
  LessThanLessThanEqualsToken = 71,
  GreaterThanGreaterThanEqualsToken = 72,
  GreaterThanGreaterThanGreaterThanEqualsToken = 73,
  AmpersandEqualsToken = 74,
  BarEqualsToken = 75,
  BarBarEqualsToken = 76,
  AmpersandAmpersandEqualsToken = 77,
  QuestionQuestionEqualsToken = 78,
  CaretEqualsToken = 79,
  Identifier = 80,
  PrivateIdentifier = 81,
  // Keywords
  BreakKeyword = 83,
  CaseKeyword = 84,
  CatchKeyword = 85,
  ClassKeyword = 86,
  ConstKeyword = 87,
  ContinueKeyword = 88,
  DebuggerKeyword = 89,
  DefaultKeyword = 90,
  DeleteKeyword = 91,
  DoKeyword = 92,
  ElseKeyword = 93,
  EnumKeyword = 94,
  ExportKeyword = 95,
  ExtendsKeyword = 96,
  FalseKeyword = 97,
  FinallyKeyword = 98,
  ForKeyword = 99,
  FunctionKeyword = 100,
  IfKeyword = 101,
  ImportKeyword = 102,
  InKeyword = 103,
  InstanceOfKeyword = 104,
  NewKeyword = 105,
  NullKeyword = 106,
  ReturnKeyword = 107,
  SuperKeyword = 108,
  SwitchKeyword = 109,
  ThisKeyword = 110,
  ThrowKeyword = 111,
  TrueKeyword = 112,
  TryKeyword = 113,
  TypeOfKeyword = 114,
  VarKeyword = 115,
  VoidKeyword = 116,
  WhileKeyword = 117,
  WithKeyword = 118,
  // TypeScript keywords
  ImplementsKeyword = 119,
  InterfaceKeyword = 120,
  LetKeyword = 121,
  PackageKeyword = 122,
  PrivateKeyword = 123,
  ProtectedKeyword = 124,
  PublicKeyword = 125,
  StaticKeyword = 126,
  YieldKeyword = 127,
  AbstractKeyword = 128,
  AccessorKeyword = 129,
  AsKeyword = 130,
  AssertsKeyword = 131,
  AssertKeyword = 132,
  AnyKeyword = 133,
  AsyncKeyword = 134,
  AwaitKeyword = 135,
  BooleanKeyword = 136,
  ConstructorKeyword = 137,
  DeclareKeyword = 138,
  GetKeyword = 139,
  InferKeyword = 140,
  IntrinsicKeyword = 141,
  IsKeyword = 142,
  KeyOfKeyword = 143,
  ModuleKeyword = 144,
  NamespaceKeyword = 145,
  NeverKeyword = 146,
  OutKeyword = 147,
  ReadonlyKeyword = 148,
  RequireKeyword = 149,
  NumberKeyword = 150,
  ObjectKeyword = 151,
  SatisfiesKeyword = 152,
  SetKeyword = 153,
  StringKeyword = 154,
  SymbolKeyword = 155,
  TypeKeyword = 156,
  UndefinedKeyword = 157,
  UniqueKeyword = 158,
  UnknownKeyword = 159,
  UsingKeyword = 160,
  FromKeyword = 161,
  GlobalKeyword = 162,
  BigIntKeyword = 163,
  OverrideKeyword = 164,
  OfKeyword = 165,
  DeferKeyword = 166,
  QualifiedName = 167,
  ComputedPropertyName = 168,
  TypeParameter = 169,
  Parameter = 170,
  Decorator = 171,
  PropertySignature = 172,
  PropertyDeclaration = 173,
  MethodSignature = 174,
  MethodDeclaration = 175,
  ClassStaticBlockDeclaration = 176,
  Constructor = 177,
  GetAccessor = 178,
  SetAccessor = 179,
  CallSignature = 180,
  ConstructSignature = 181,
  IndexSignature = 182,
  TypePredicate = 183,
  TypeReference = 184,
  FunctionType = 185,
  ConstructorType = 186,
  TypeQuery = 187,
  TypeLiteral = 188,
  ArrayType = 189,
  TupleType = 190,
  OptionalType = 191,
  RestType = 192,
  UnionType = 193,
  IntersectionType = 194,
  ConditionalType = 195,
  InferType = 196,
  ParenthesizedType = 197,
  ThisType = 198,
  TypeOperator = 199,
  IndexedAccessType = 200,
  MappedType = 201,
  LiteralType = 202,
  NamedTupleMember = 203,
  TemplateLiteralType = 204,
  TemplateLiteralTypeSpan = 205,
  ImportType = 206,
  ObjectBindingPattern = 207,
  ArrayBindingPattern = 208,
  BindingElement = 209,
  ArrayLiteralExpression = 210,
  ObjectLiteralExpression = 211,
  PropertyAccessExpression = 212,
  ElementAccessExpression = 213,
  CallExpression = 214,
  NewExpression = 215,
  TaggedTemplateExpression = 216,
  TypeAssertionExpression = 217,
  ParenthesizedExpression = 218,
  FunctionExpression = 219,
  ArrowFunction = 220,
  DeleteExpression = 221,
  TypeOfExpression = 222,
  VoidExpression = 223,
  AwaitExpression = 224,
  PrefixUnaryExpression = 225,
  PostfixUnaryExpression = 226,
  BinaryExpression = 227,
  ConditionalExpression = 228,
  TemplateExpression = 229,
  YieldExpression = 230,
  SpreadElement = 231,
  ClassExpression = 232,
  OmittedExpression = 233,
  ExpressionWithTypeArguments = 234,
  AsExpression = 235,
  NonNullExpression = 236,
  MetaProperty = 237,
  SyntheticExpression = 238,
  SatisfiesExpression = 239,
  TemplateSpan = 240,
  SemicolonClassElement = 241,
  Block = 242,
  EmptyStatement = 243,
  VariableStatement = 244,
  ExpressionStatement = 245,
  IfStatement = 246,
  DoStatement = 247,
  WhileStatement = 248,
  ForStatement = 249,
  ForInStatement = 250,
  ForOfStatement = 251,
  ContinueStatement = 252,
  BreakStatement = 253,
  ReturnStatement = 254,
  WithStatement = 255,
  SwitchStatement = 256,
  LabeledStatement = 257,
  ThrowStatement = 258,
  TryStatement = 259,
  DebuggerStatement = 260,
  VariableDeclaration = 261,
  VariableDeclarationList = 262,
  FunctionDeclaration = 263,
  ClassDeclaration = 264,
  InterfaceDeclaration = 265,
  TypeAliasDeclaration = 266,
  EnumDeclaration = 267,
  ModuleDeclaration = 268,
  ModuleBlock = 269,
  CaseBlock = 270,
  NamespaceExportDeclaration = 271,
  ImportEqualsDeclaration = 272,
  ImportDeclaration = 273,
  ImportClause = 274,
  NamespaceImport = 275,
  NamedImports = 276,
  ImportSpecifier = 277,
  ExportAssignment = 278,
  ExportDeclaration = 279,
  NamedExports = 280,
  NamespaceExport = 281,
  ExportSpecifier = 282,
  MissingDeclaration = 283,
  ExternalModuleReference = 284,
  JsxElement = 285,
  JsxSelfClosingElement = 286,
  JsxOpeningElement = 287,
  JsxClosingElement = 288,
  JsxFragment = 289,
  JsxOpeningFragment = 290,
  JsxClosingFragment = 291,
  JsxAttribute = 292,
  JsxAttributes = 293,
  JsxSpreadAttribute = 294,
  JsxExpression = 295,
  JsxNamespacedName = 296,
  CaseClause = 297,
  DefaultClause = 298,
  HeritageClause = 299,
  CatchClause = 300,
  ImportAttributes = 301,
  ImportAttribute = 302,
  PropertyAssignment = 304,
  ShorthandPropertyAssignment = 305,
  SpreadAssignment = 306,
  EnumMember = 307,
  SourceFile = 308,
  Bundle = 309,
  JSDocTypeExpression = 310,
  JSDocNameReference = 311,
  JSDocMemberName = 312,
  JSDocAllType = 313,
  JSDocUnknownType = 314,
  JSDocNullableType = 315,
  JSDocNonNullableType = 316,
  JSDocOptionalType = 317,
  JSDocFunctionType = 318,
  JSDocVariadicType = 319,
  JSDocNamepathType = 320,
  JSDoc = 321,
  JSDocText = 322,
  JSDocTypeLiteral = 323,
  JSDocSignature = 324,
  JSDocLink = 325,
  JSDocLinkCode = 326,
  JSDocLinkPlain = 327,
  JSDocTag = 328,
  JSDocAugmentsTag = 329,
  JSDocImplementsTag = 330,
  JSDocAuthorTag = 331,
  JSDocDeprecatedTag = 332,
  JSDocClassTag = 333,
  JSDocPublicTag = 334,
  JSDocPrivateTag = 335,
  JSDocProtectedTag = 336,
  JSDocReadonlyTag = 337,
  JSDocOverrideTag = 338,
  JSDocCallbackTag = 339,
  JSDocOverloadTag = 340,
  JSDocEnumTag = 341,
  JSDocParameterTag = 342,
  JSDocReturnTag = 343,
  JSDocThisTag = 344,
  JSDocTypeTag = 345,
  JSDocTemplateTag = 346,
  JSDocTypedefTag = 347,
  JSDocSeeTag = 348,
  JSDocPropertyTag = 349,
  JSDocThrowsTag = 350,
  JSDocSatisfiesTag = 351,
  JSDocImportTag = 352,
  SyntaxList = 353,
  NotEmittedStatement = 354,
  NotEmittedTypeElement = 355,
  PartiallyEmittedExpression = 356,
  CommaListExpression = 357,
  SyntheticReferenceExpression = 358,
  Count = 359,

  // Range markers for token classification
  FirstAssignment = 64,
  LastAssignment = 79,
  FirstCompoundAssignment = 65,
  LastCompoundAssignment = 79,
  FirstReservedWord = 83,
  LastReservedWord = 118,
  FirstKeyword = 83,
  LastKeyword = 166,
  FirstFutureReservedWord = 119,
  LastFutureReservedWord = 127,
  FirstTypeNode = 183,
  LastTypeNode = 206,
  FirstPunctuation = 19,
  LastPunctuation = 79,
  FirstToken = 0,
  LastToken = 166,
  FirstTriviaToken = 2,
  LastTriviaToken = 7,
  FirstLiteralToken = 9,
  LastLiteralToken = 15,
  FirstTemplateToken = 15,
  LastTemplateToken = 18,
  FirstBinaryOperator = 30,
  LastBinaryOperator = 79,
  FirstStatement = 244,
  LastStatement = 260,
  FirstNode = 167,
  FirstJSDocNode = 310,
  LastJSDocNode = 352,
  FirstJSDocTagNode = 328,
  LastJSDocTagNode = 352
}
```

### Base Node Interface

Core node interface inherited by all AST nodes.

```typescript { .api }
interface Node {
  readonly kind: SyntaxKind;
  readonly flags: NodeFlags;
  readonly parent: Node;
  
  getSourceFile(): SourceFile;
  getChildCount(sourceFile?: SourceFile): number;
  getChildAt(index: number, sourceFile?: SourceFile): Node;
  getChildren(sourceFile?: SourceFile): readonly Node[];
  getStart(sourceFile?: SourceFile, includeJsDocComment?: boolean): number;
  getFullStart(): number;
  getEnd(): number;
  getWidth(sourceFile?: SourceFileLike): number;
  getFullWidth(): number;
  getLeadingTriviaWidth(sourceFile?: SourceFile): number;
  getFullText(sourceFile?: SourceFile): string;
  getText(sourceFile?: SourceFile): string;
  getFirstToken(sourceFile?: SourceFile): Node | undefined;
  getLastToken(sourceFile?: SourceFile): Node | undefined;
  forEachChild<T>(
    cbNode: (node: Node) => T | undefined,
    cbNodes?: (nodes: NodeArray<Node>) => T | undefined
  ): T | undefined;
}
```

### Node Traversal

Traverse AST nodes recursively.

```typescript { .api }
function forEachChild<T>(
  node: Node,
  cbNode: (node: Node) => T | undefined,
  cbNodes?: (nodes: NodeArray<Node>) => T | undefined
): T | undefined;

function visitNode<T extends Node>(
  node: T | undefined,
  visitor: Visitor<T>,
  test?: (node: Node) => boolean,
  lift?: (node: readonly Node[]) => T
): T;

function visitNodes<T extends Node>(
  nodes: NodeArray<T> | undefined,
  visitor: Visitor<T>,
  test?: (node: Node) => boolean,
  start?: number,
  count?: number
): NodeArray<T>;

function visitEachChild<T extends Node>(
  node: T,
  visitor: Visitor,
  context: TransformationContext
): T;

type Visitor<TIn extends Node = Node, TOut extends Node = Node> = (
  node: TIn
) => VisitResult<TOut>;

type VisitResult<T extends Node> = T | T[] | undefined;
```

### Statement Nodes

```typescript { .api }
interface Statement extends Node {
  _statementBrand: any;
}

interface EmptyStatement extends Statement {
  kind: SyntaxKind.EmptyStatement;
}

interface VariableStatement extends Statement {
  kind: SyntaxKind.VariableStatement;
  modifiers?: NodeArray<ModifierLike>;
  declarationList: VariableDeclarationList;
}

interface ExpressionStatement extends Statement {
  kind: SyntaxKind.ExpressionStatement;
  expression: Expression;
}

interface IfStatement extends Statement {
  kind: SyntaxKind.IfStatement;
  expression: Expression;
  thenStatement: Statement;
  elseStatement?: Statement;
}

interface DoStatement extends Statement {
  kind: SyntaxKind.DoStatement;
  statement: Statement;
  expression: Expression;
}

interface WhileStatement extends Statement {
  kind: SyntaxKind.WhileStatement;
  expression: Expression;
  statement: Statement;
}

interface ForStatement extends Statement {
  kind: SyntaxKind.ForStatement;
  initializer?: ForInitializer;
  condition?: Expression;
  incrementor?: Expression;
  statement: Statement;
}

interface ForInStatement extends Statement {
  kind: SyntaxKind.ForInStatement;
  initializer: ForInitializer;
  expression: Expression;
  statement: Statement;
}

interface ForOfStatement extends Statement {
  kind: SyntaxKind.ForOfStatement;
  awaitModifier?: AwaitKeyword;
  initializer: ForInitializer;
  expression: Expression;
  statement: Statement;
}

interface BreakStatement extends Statement {
  kind: SyntaxKind.BreakStatement;
  label?: Identifier;
}

interface ContinueStatement extends Statement {
  kind: SyntaxKind.ContinueStatement;
  label?: Identifier;
}

interface ReturnStatement extends Statement {
  kind: SyntaxKind.ReturnStatement;
  expression?: Expression;
}

interface SwitchStatement extends Statement {
  kind: SyntaxKind.SwitchStatement;
  expression: Expression;
  caseBlock: CaseBlock;
}

interface ThrowStatement extends Statement {
  kind: SyntaxKind.ThrowStatement;
  expression: Expression;
}

interface TryStatement extends Statement {
  kind: SyntaxKind.TryStatement;
  tryBlock: Block;
  catchClause?: CatchClause;
  finallyBlock?: Block;
}
```

### Expression Nodes

```typescript { .api }
interface Expression extends Node {
  _expressionBrand: any;
}

interface BinaryExpression extends Expression {
  kind: SyntaxKind.BinaryExpression;
  left: Expression;
  operatorToken: BinaryOperatorToken;
  right: Expression;
}

interface CallExpression extends LeftHandSideExpression {
  kind: SyntaxKind.CallExpression;
  expression: LeftHandSideExpression;
  typeArguments?: NodeArray<TypeNode>;
  arguments: NodeArray<Expression>;
}

interface PropertyAccessExpression extends MemberExpression {
  kind: SyntaxKind.PropertyAccessExpression;
  expression: LeftHandSideExpression;
  questionDotToken?: QuestionDotToken;
  name: Identifier | PrivateIdentifier;
}

interface ElementAccessExpression extends MemberExpression {
  kind: SyntaxKind.ElementAccessExpression;
  expression: LeftHandSideExpression;
  questionDotToken?: QuestionDotToken;
  argumentExpression: Expression;
}

interface ArrayLiteralExpression extends PrimaryExpression {
  kind: SyntaxKind.ArrayLiteralExpression;
  elements: NodeArray<Expression>;
}

interface ObjectLiteralExpression extends PrimaryExpression {
  kind: SyntaxKind.ObjectLiteralExpression;
  properties: NodeArray<ObjectLiteralElementLike>;
}

interface ArrowFunction extends Expression, FunctionLikeDeclarationBase {
  kind: SyntaxKind.ArrowFunction;
  equalsGreaterThanToken: EqualsGreaterThanToken;
  body: ConciseBody;
}

interface ConditionalExpression extends Expression {
  kind: SyntaxKind.ConditionalExpression;
  condition: Expression;
  questionToken: QuestionToken;
  whenTrue: Expression;
  colonToken: ColonToken;
  whenFalse: Expression;
}
```

### Declaration Nodes

```typescript { .api }
interface Declaration extends Node {
  _declarationBrand: any;
}

interface FunctionDeclaration extends FunctionLikeDeclarationBase, DeclarationStatement {
  kind: SyntaxKind.FunctionDeclaration;
  modifiers?: NodeArray<ModifierLike>;
  name?: Identifier;
  body?: FunctionBody;
}

interface ClassDeclaration extends ClassLikeDeclarationBase, DeclarationStatement {
  kind: SyntaxKind.ClassDeclaration;
  modifiers?: NodeArray<ModifierLike>;
  name?: Identifier;
}

interface InterfaceDeclaration extends DeclarationStatement, JSDocContainer {
  kind: SyntaxKind.InterfaceDeclaration;
  modifiers?: NodeArray<ModifierLike>;
  name: Identifier;
  typeParameters?: NodeArray<TypeParameterDeclaration>;
  heritageClauses?: NodeArray<HeritageClause>;
  members: NodeArray<TypeElement>;
}

interface TypeAliasDeclaration extends DeclarationStatement, JSDocContainer {
  kind: SyntaxKind.TypeAliasDeclaration;
  modifiers?: NodeArray<ModifierLike>;
  name: Identifier;
  typeParameters?: NodeArray<TypeParameterDeclaration>;
  type: TypeNode;
}

interface EnumDeclaration extends DeclarationStatement, JSDocContainer {
  kind: SyntaxKind.EnumDeclaration;
  modifiers?: NodeArray<ModifierLike>;
  name: Identifier;
  members: NodeArray<EnumMember>;
}

interface VariableDeclaration extends NamedDeclaration, JSDocContainer {
  kind: SyntaxKind.VariableDeclaration;
  parent: VariableDeclarationList | CatchClause;
  name: BindingName;
  exclamationToken?: ExclamationToken;
  type?: TypeNode;
  initializer?: Expression;
}
```

### Type Nodes

```typescript { .api }
interface TypeNode extends Node {
  _typeNodeBrand: any;
}

interface TypeReferenceNode extends TypeNode {
  kind: SyntaxKind.TypeReference;
  typeName: EntityName;
  typeArguments?: NodeArray<TypeNode>;
}

interface UnionTypeNode extends TypeNode {
  kind: SyntaxKind.UnionType;
  types: NodeArray<TypeNode>;
}

interface IntersectionTypeNode extends TypeNode {
  kind: SyntaxKind.IntersectionType;
  types: NodeArray<TypeNode>;
}

interface ConditionalTypeNode extends TypeNode {
  kind: SyntaxKind.ConditionalType;
  checkType: TypeNode;
  extendsType: TypeNode;
  trueType: TypeNode;
  falseType: TypeNode;
}

interface ArrayTypeNode extends TypeNode {
  kind: SyntaxKind.ArrayType;
  elementType: TypeNode;
}

interface TupleTypeNode extends TypeNode {
  kind: SyntaxKind.TupleType;
  elements: NodeArray<TypeNode | NamedTupleMember>;
}

interface FunctionTypeNode extends TypeNode, SignatureDeclarationBase {
  kind: SyntaxKind.FunctionType;
}

interface MappedTypeNode extends TypeNode, Declaration {
  kind: SyntaxKind.MappedType;
  readonlyToken?: ReadonlyKeyword | PlusToken | MinusToken;
  typeParameter: TypeParameterDeclaration;
  nameType?: TypeNode;
  questionToken?: QuestionToken | PlusToken | MinusToken;
  type?: TypeNode;
  members?: NodeArray<TypeElement>;
}
```

### Specialized Type Aliases

Common type aliases for node categories.

```typescript { .api }
/**
 * Identifier or qualified name (e.g., Foo.Bar)
 */
type EntityName = Identifier | QualifiedName;

/**
 * Property name in object literals and class members
 */
type PropertyName =
  | Identifier
  | StringLiteral
  | NoSubstitutionTemplateLiteral
  | NumericLiteral
  | ComputedPropertyName
  | PrivateIdentifier
  | BigIntLiteral;

/**
 * Member name (public or private identifier)
 */
type MemberName = Identifier | PrivateIdentifier;

/**
 * Name of a declaration (property, binding, etc.)
 */
type DeclarationName =
  | PropertyName
  | JsxAttributeName
  | StringLiteralLike
  | ElementAccessExpression
  | BindingPattern
  | EntityNameExpression;

/**
 * Expression that can be used as an entity name
 */
type EntityNameExpression = Identifier | PropertyAccessEntityNameExpression;

/**
 * Entity name or entity name expression
 */
type EntityNameOrEntityNameExpression = EntityName | EntityNameExpression;

/**
 * Nodes that can have a type annotation
 */
type HasType =
  | SignatureDeclaration
  | VariableDeclaration
  | ParameterDeclaration
  | PropertySignature
  | PropertyDeclaration
  | TypePredicateNode
  | ParenthesizedTypeNode
  | TypeOperatorNode
  | MappedTypeNode
  | AssertionExpression
  | TypeAliasDeclaration
  | JSDocTypeExpression
  | JSDocNonNullableType
  | JSDocNullableType
  | JSDocOptionalType
  | JSDocVariadicType;

/**
 * Nodes that can have JSDoc comments
 */
type HasJSDoc =
  | ParameterDeclaration
  | ClassDeclaration
  | ConstructorDeclaration
  | MethodDeclaration
  | PropertyDeclaration
  | ArrowFunction
  | ParenthesizedExpression
  | SpreadAssignment
  | ShorthandPropertyAssignment
  | PropertyAssignment
  | FunctionExpression
  | EmptyStatement
  | DebuggerStatement
  | Block
  | VariableStatement
  | ExpressionStatement
  | IfStatement
  | DoStatement
  | WhileStatement
  | ForStatement
  | ForInStatement
  | ForOfStatement
  | BreakStatement
  | ContinueStatement
  | ReturnStatement
  | WithStatement
  | SwitchStatement
  | LabeledStatement
  | ThrowStatement
  | TryStatement
  | FunctionDeclaration
  | ConstructSignatureDeclaration
  | MethodSignature
  | CallSignatureDeclaration
  | IndexSignatureDeclaration
  | TypeAliasDeclaration
  | InterfaceDeclaration
  | ModuleDeclaration
  | ImportEqualsDeclaration
  | ImportDeclaration
  | NamespaceExportDeclaration
  | ExportAssignment
  | ExportDeclaration
  | NamedTupleMember
  | EndOfFileToken;

/**
 * Nodes that can have decorators
 */
type HasDecorators =
  | ParameterDeclaration
  | PropertyDeclaration
  | MethodDeclaration
  | GetAccessorDeclaration
  | SetAccessorDeclaration
  | ClassExpression
  | ClassDeclaration;
```

### Node Flags

```typescript { .api }
enum NodeFlags {
  None = 0,
  Let = 1,
  Const = 2,
  Using = 4,
  NestedNamespace = 8,
  Synthesized = 16,
  Namespace = 32,
  OptionalChain = 64,
  ExportContext = 128,
  ContainsThis = 256,
  HasImplicitReturn = 512,
  HasExplicitReturn = 1024,
  GlobalAugmentation = 2048,
  HasAsyncFunctions = 4096,
  DisallowInContext = 8192,
  YieldContext = 16384,
  DecoratorContext = 32768,
  AwaitContext = 65536,
  ThisNodeHasError = 262144,
  JavaScriptFile = 524288,
  ThisNodeOrAnySubNodesHasError = 1048576,
  HasAggregatedChildData = 2097152,
  JSDoc = 16777216,
  JsonFile = 134217728
}
```

### JSDoc Utilities

Functions for extracting and working with JSDoc comments, tags, and type annotations on AST nodes.

```typescript { .api }
/**
 * Get all JSDoc tags related to a node, including those on parent nodes
 * @param node - AST node to get JSDoc tags for
 * @returns Array of JSDoc tags
 */
function getJSDocTags(node: Node): readonly JSDocTag[];

/**
 * Get all JSDoc tags that match a specified predicate
 * @param node - AST node to search
 * @param predicate - Type guard predicate to filter tags
 * @returns Filtered array of JSDoc tags
 */
function getAllJSDocTags<T extends JSDocTag>(
  node: Node,
  predicate: (tag: JSDocTag) => tag is T
): readonly T[];

/**
 * Get all JSDoc tags of a specified syntax kind
 * @param node - AST node to search
 * @param kind - SyntaxKind to filter by
 * @returns Array of JSDoc tags of the specified kind
 */
function getAllJSDocTagsOfKind(
  node: Node,
  kind: SyntaxKind
): readonly JSDocTag[];

/**
 * Get JSDoc parameter tags for a parameter declaration
 * @param param - Parameter declaration node
 * @returns Array of JSDoc parameter tags
 * @remarks For binding patterns, parameter tags are matched by position
 */
function getJSDocParameterTags(
  param: ParameterDeclaration
): readonly JSDocParameterTag[];

/**
 * Get JSDoc type parameter tags for a type parameter declaration
 * @param param - Type parameter declaration
 * @returns Array of JSDoc template tags
 */
function getJSDocTypeParameterTags(
  param: TypeParameterDeclaration
): readonly JSDocTemplateTag[];

/**
 * Check if a node has JSDoc parameter tags
 * @param node - Function-like or signature declaration
 * @returns True if the node has JSDoc parameter tags
 */
function hasJSDocParameterTags(
  node: FunctionLikeDeclaration | SignatureDeclaration
): boolean;

/**
 * Get the type node for a node if provided via JSDoc
 * @param node - AST node
 * @returns Type node or undefined
 */
function getJSDocType(node: Node): TypeNode | undefined;

/**
 * Get the return type node for a node if provided via JSDoc
 * @param node - AST node
 * @returns Return type node or undefined
 */
function getJSDocReturnType(node: Node): TypeNode | undefined;

/**
 * Get the text of a JSDoc comment, flattening links to their text
 * @param comment - JSDoc comment string or node array
 * @returns Flattened comment text or undefined
 */
function getTextOfJSDocComment(
  comment?: string | NodeArray<JSDocComment>
): string | undefined;

// Specific JSDoc tag getters
function getJSDocTypeTag(node: Node): JSDocTypeTag | undefined;
function getJSDocReturnTag(node: Node): JSDocReturnTag | undefined;
function getJSDocTemplateTag(node: Node): JSDocTemplateTag | undefined;
function getJSDocThisTag(node: Node): JSDocThisTag | undefined;
function getJSDocEnumTag(node: Node): JSDocEnumTag | undefined;
function getJSDocClassTag(node: Node): JSDocClassTag | undefined;
function getJSDocPublicTag(node: Node): JSDocPublicTag | undefined;
function getJSDocPrivateTag(node: Node): JSDocPrivateTag | undefined;
function getJSDocProtectedTag(node: Node): JSDocProtectedTag | undefined;
function getJSDocReadonlyTag(node: Node): JSDocReadonlyTag | undefined;
function getJSDocOverrideTagNoCache(node: Node): JSDocOverrideTag | undefined;
function getJSDocDeprecatedTag(node: Node): JSDocDeprecatedTag | undefined;
function getJSDocSatisfiesTag(node: Node): JSDocSatisfiesTag | undefined;
function getJSDocAugmentsTag(node: Node): JSDocAugmentsTag | undefined;
function getJSDocImplementsTags(node: Node): readonly JSDocImplementsTag[];
```

**Usage Example:**

```typescript
import * as ts from 'typescript';

const sourceFile = ts.createSourceFile(
  'example.ts',
  `
  /**
   * Adds two numbers together
   * @param a - First number
   * @param b - Second number
   * @returns The sum of a and b
   * @deprecated Use the new add() function instead
   */
  function oldAdd(a: number, b: number): number {
    return a + b;
  }
  `,
  ts.ScriptTarget.Latest,
  true
);

// Find the function declaration
const functionDecl = sourceFile.statements[0] as ts.FunctionDeclaration;

// Get all JSDoc tags
const allTags = ts.getJSDocTags(functionDecl);
console.log(`Found ${allTags.length} JSDoc tags`);

// Get specific tags
const deprecatedTag = ts.getJSDocDeprecatedTag(functionDecl);
if (deprecatedTag) {
  console.log('Function is deprecated');
}

const returnTag = ts.getJSDocReturnTag(functionDecl);
if (returnTag && returnTag.comment) {
  const commentText = ts.getTextOfJSDocComment(returnTag.comment);
  console.log(`Return: ${commentText}`);
}

// Get parameter tags
const params = functionDecl.parameters;
if (params.length > 0) {
  const paramTags = ts.getJSDocParameterTags(params[0]);
  console.log(`Parameter 'a' has ${paramTags.length} JSDoc tags`);
}
```

### Type Guards

Comprehensive type guard functions for all node types.

```typescript { .api }
function isSourceFile(node: Node): node is SourceFile;
function isIdentifier(node: Node): node is Identifier;
function isQualifiedName(node: Node): node is QualifiedName;
function isComputedPropertyName(node: Node): node is ComputedPropertyName;
function isPrivateIdentifier(node: Node): node is PrivateIdentifier;
function isTypeParameterDeclaration(node: Node): node is TypeParameterDeclaration;
function isParameter(node: Node): node is ParameterDeclaration;
function isDecorator(node: Node): node is Decorator;
function isPropertySignature(node: Node): node is PropertySignature;
function isPropertyDeclaration(node: Node): node is PropertyDeclaration;
function isMethodSignature(node: Node): node is MethodSignature;
function isMethodDeclaration(node: Node): node is MethodDeclaration;
function isClassStaticBlockDeclaration(node: Node): node is ClassStaticBlockDeclaration;
function isConstructorDeclaration(node: Node): node is ConstructorDeclaration;
function isGetAccessorDeclaration(node: Node): node is GetAccessorDeclaration;
function isSetAccessorDeclaration(node: Node): node is SetAccessorDeclaration;
function isCallSignatureDeclaration(node: Node): node is CallSignatureDeclaration;
function isConstructSignatureDeclaration(node: Node): node is ConstructSignatureDeclaration;
function isIndexSignatureDeclaration(node: Node): node is IndexSignatureDeclaration;
function isTypePredicateNode(node: Node): node is TypePredicateNode;
function isTypeReferenceNode(node: Node): node is TypeReferenceNode;
function isFunctionTypeNode(node: Node): node is FunctionTypeNode;
function isConstructorTypeNode(node: Node): node is ConstructorTypeNode;
function isTypeQueryNode(node: Node): node is TypeQueryNode;
function isTypeLiteralNode(node: Node): node is TypeLiteralNode;
function isArrayTypeNode(node: Node): node is ArrayTypeNode;
function isTupleTypeNode(node: Node): node is TupleTypeNode;
function isUnionTypeNode(node: Node): node is UnionTypeNode;
function isIntersectionTypeNode(node: Node): node is IntersectionTypeNode;
function isConditionalTypeNode(node: Node): node is ConditionalTypeNode;
function isInferTypeNode(node: Node): node is InferTypeNode;
function isParenthesizedTypeNode(node: Node): node is ParenthesizedTypeNode;
function isThisTypeNode(node: Node): node is ThisTypeNode;
function isTypeOperatorNode(node: Node): node is TypeOperatorNode;
function isIndexedAccessTypeNode(node: Node): node is IndexedAccessTypeNode;
function isMappedTypeNode(node: Node): node is MappedTypeNode;
function isLiteralTypeNode(node: Node): node is LiteralTypeNode;
function isBooleanLiteral(node: Node): node is BooleanLiteral;
function isFunctionExpression(node: Node): node is FunctionExpression;
function isArrowFunction(node: Node): node is ArrowFunction;
function isArrayLiteralExpression(node: Node): node is ArrayLiteralExpression;
function isObjectLiteralExpression(node: Node): node is ObjectLiteralExpression;
function isPropertyAccessExpression(node: Node): node is PropertyAccessExpression;
function isElementAccessExpression(node: Node): node is ElementAccessExpression;
function isCallExpression(node: Node): node is CallExpression;
function isNewExpression(node: Node): node is NewExpression;
function isTaggedTemplateExpression(node: Node): node is TaggedTemplateExpression;
function isTypeAssertionExpression(node: Node): node is TypeAssertion;
function isParenthesizedExpression(node: Node): node is ParenthesizedExpression;
function isDeleteExpression(node: Node): node is DeleteExpression;
function isTypeOfExpression(node: Node): node is TypeOfExpression;
function isVoidExpression(node: Node): node is VoidExpression;
function isAwaitExpression(node: Node): node is AwaitExpression;
function isPrefixUnaryExpression(node: Node): node is PrefixUnaryExpression;
function isPostfixUnaryExpression(node: Node): node is PostfixUnaryExpression;
function isBinaryExpression(node: Node): node is BinaryExpression;
function isConditionalExpression(node: Node): node is ConditionalExpression;
function isTemplateExpression(node: Node): node is TemplateExpression;
function isYieldExpression(node: Node): node is YieldExpression;
function isSpreadElement(node: Node): node is SpreadElement;
function isClassExpression(node: Node): node is ClassExpression;
function isOmittedExpression(node: Node): node is OmittedExpression;
function isExpressionWithTypeArguments(node: Node): node is ExpressionWithTypeArguments;
function isAsExpression(node: Node): node is AsExpression;
function isNonNullExpression(node: Node): node is NonNullExpression;
function isSatisfiesExpression(node: Node): node is SatisfiesExpression;
function isBlock(node: Node): node is Block;
function isVariableStatement(node: Node): node is VariableStatement;
function isEmptyStatement(node: Node): node is EmptyStatement;
function isExpressionStatement(node: Node): node is ExpressionStatement;
function isIfStatement(node: Node): node is IfStatement;
function isDoStatement(node: Node): node is DoStatement;
function isWhileStatement(node: Node): node is WhileStatement;
function isForStatement(node: Node): node is ForStatement;
function isForInStatement(node: Node): node is ForInStatement;
function isForOfStatement(node: Node): node is ForOfStatement;
function isContinueStatement(node: Node): node is ContinueStatement;
function isBreakStatement(node: Node): node is BreakStatement;
function isReturnStatement(node: Node): node is ReturnStatement;
function isWithStatement(node: Node): node is WithStatement;
function isSwitchStatement(node: Node): node is SwitchStatement;
function isLabeledStatement(node: Node): node is LabeledStatement;
function isThrowStatement(node: Node): node is ThrowStatement;
function isTryStatement(node: Node): node is TryStatement;
function isDebuggerStatement(node: Node): node is DebuggerStatement;
function isVariableDeclaration(node: Node): node is VariableDeclaration;
function isVariableDeclarationList(node: Node): node is VariableDeclarationList;
function isFunctionDeclaration(node: Node): node is FunctionDeclaration;
function isClassDeclaration(node: Node): node is ClassDeclaration;
function isInterfaceDeclaration(node: Node): node is InterfaceDeclaration;
function isTypeAliasDeclaration(node: Node): node is TypeAliasDeclaration;
function isEnumDeclaration(node: Node): node is EnumDeclaration;
function isModuleDeclaration(node: Node): node is ModuleDeclaration;
function isImportEqualsDeclaration(node: Node): node is ImportEqualsDeclaration;
function isImportDeclaration(node: Node): node is ImportDeclaration;
function isExportAssignment(node: Node): node is ExportAssignment;
function isExportDeclaration(node: Node): node is ExportDeclaration;
```
