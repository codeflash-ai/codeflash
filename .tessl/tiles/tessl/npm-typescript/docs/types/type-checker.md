# Type System

TypeChecker API for type inference, analysis, and symbol resolution. The type system provides 150+ methods for analyzing TypeScript types and symbols.

## Capabilities

### TypeChecker Interface

Primary interface for type analysis and symbol resolution.

```typescript { .api }
interface TypeChecker {
  /* Type Operations */
  getTypeAtLocation(node: Node): Type;
  getDeclaredTypeOfSymbol(symbol: Symbol): Type;
  getTypeOfSymbolAtLocation(symbol: Symbol, node: Node): Type;
  getTypeFromTypeNode(node: TypeNode): Type;
  getApparentType(type: Type): Type;
  getBaseTypeOfLiteralType(type: Type): Type;
  getWidenedType(type: Type): Type;
  getNullableType(type: Type, flags: TypeFlags): Type;
  getNonNullableType(type: Type): Type;
  getTypePredicateOfSignature(signature: Signature): TypePredicate | undefined;
  getReturnTypeOfSignature(signature: Signature): Type;
  
  /* Type Properties */
  getPropertiesOfType(type: Type): Symbol[];
  getPropertyOfType(type: Type, propertyName: string): Symbol | undefined;
  getIndexInfoOfType(type: Type, kind: IndexKind): IndexInfo | undefined;
  getIndexInfosOfType(type: Type): readonly IndexInfo[];
  getSignaturesOfType(type: Type, kind: SignatureKind): readonly Signature[];
  getIndexTypeOfType(type: Type, kind: IndexKind): Type | undefined;
  getBaseTypes(type: InterfaceType): BaseType[];
  
  /* Type Checking */
  isTypeAssignableTo(source: Type, target: Type): boolean;

  /* Symbol Operations */
  getSymbolAtLocation(node: Node): Symbol | undefined;
  getSymbolsInScope(location: Node, meaning: SymbolFlags): Symbol[];
  getTypeOfSymbol(symbol: Symbol): Type;
  getSymbolsOfParameterPropertyDeclaration(parameter: ParameterDeclaration, parameterName: string): Symbol[];
  getShorthandAssignmentValueSymbol(location: Node | undefined): Symbol | undefined;
  getExportSpecifierLocalTargetSymbol(location: ExportSpecifier | Identifier): Symbol | undefined;
  getExportSymbolOfSymbol(symbol: Symbol): Symbol;
  getPropertySymbolOfDestructuringAssignment(location: Identifier): Symbol | undefined;
  getTypeOfAssignmentPattern(pattern: AssignmentPattern): Type;
  getPrivateIdentifierPropertyOfType(leftType: Type, name: string, location: Node): Symbol | undefined;
  getIndexInfosOfIndexSymbol(indexSymbol: Symbol, siblingSymbols?: Symbol[]): IndexInfo[];
  getAwaitedType(type: Type): Type | undefined;
  getImmediateAliasedSymbol(symbol: Symbol): Symbol | undefined;
  getAugmentedPropertiesOfType(type: Type): Symbol[];
  getRootSymbols(symbol: Symbol): readonly Symbol[];
  getSymbolOfExpando(node: Node, allowDeclaration: boolean): Symbol | undefined;
  isUndefinedSymbol(symbol: Symbol): boolean;
  isArgumentsSymbol(symbol: Symbol): boolean;
  isUnknownSymbol(symbol: Symbol): boolean;
  getMergedSymbol(symbol: Symbol): Symbol;

  /* Signature Operations */
  getSignatureFromDeclaration(declaration: SignatureDeclaration): Signature | undefined;
  getResolvedSignature(node: CallLikeExpression, candidatesOutArray?: Signature[], argumentCount?: number): Signature | undefined;
  isImplementationOfOverload(node: SignatureDeclaration): boolean | undefined;
  
  /* Type String Conversion */
  typeToString(type: Type, enclosingDeclaration?: Node, flags?: TypeFormatFlags): string;
  symbolToString(symbol: Symbol, enclosingDeclaration?: Node, meaning?: SymbolFlags, flags?: SymbolFormatFlags): string;
  typePredicateToString(predicate: TypePredicate, enclosingDeclaration?: Node, flags?: TypeFormatFlags): string;
  signatureToString(signature: Signature, enclosingDeclaration?: Node, flags?: TypeFormatFlags, kind?: SignatureKind): string;
  
  /* Declaration and Module Info */
  getFullyQualifiedName(symbol: Symbol): string;
  getConstantValue(node: EnumMember | PropertyAccessExpression | ElementAccessExpression): string | number | undefined;
  isValidPropertyAccess(node: PropertyAccessExpression | QualifiedName | ImportTypeNode, propertyName: string): boolean;
  getAliasedSymbol(symbol: Symbol): Symbol;
  getExportsOfModule(moduleSymbol: Symbol): Symbol[];
  getJsxIntrinsicTagNamesAt(location: Node): Symbol[];
  isOptionalParameter(node: ParameterDeclaration): boolean;
  getAmbientModules(): Symbol[];
  tryGetMemberInModuleExports(memberName: string, moduleSymbol: Symbol): Symbol | undefined;
  getBaseConstraintOfType(type: Type): Type | undefined;
  getDefaultFromTypeParameter(type: Type): Type | undefined;
  getTypeArguments(type: TypeReference): readonly Type[];
  getContextualType(node: Expression): Type | undefined;

  /* Primitive Type Getters */
  getAnyType(): Type;
  getStringType(): Type;
  getStringLiteralType(value: string): StringLiteralType;
  getNumberType(): Type;
  getNumberLiteralType(value: number): NumberLiteralType;
  getBigIntType(): Type;
  getBigIntLiteralType(value: PseudoBigInt): BigIntLiteralType;
  getBooleanType(): Type;
  getUnknownType(): Type;
  getFalseType(): Type;
  getTrueType(): Type;
  getVoidType(): Type;
  getUndefinedType(): Type;
  getNullType(): Type;
  getESSymbolType(): Type;
  getNeverType(): Type;
  getNonPrimitiveType(): Type;

  /* Type Classification Methods */
  isArrayType(type: Type): boolean;
  isTupleType(type: Type): boolean;
  isArrayLikeType(type: Type): boolean;

  /* Advanced Type Operations */
  resolveName(name: string, location: Node | undefined, meaning: SymbolFlags, excludeGlobals: boolean): Symbol | undefined;
  runWithCancellationToken<T>(token: CancellationToken, cb: (checker: TypeChecker) => T): T;
  getTypeArgumentsForResolvedSignature(signature: Signature): readonly Type[] | undefined;

  /* AST Builder Methods */
  typeToTypeNode(type: Type, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): TypeNode | undefined;
  signatureToSignatureDeclaration(signature: Signature, kind: SyntaxKind, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined):
    | SignatureDeclaration & { typeArguments?: NodeArray<TypeNode> }
    | undefined;
  indexInfoToIndexSignatureDeclaration(indexInfo: IndexInfo, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): IndexSignatureDeclaration | undefined;
  symbolToEntityName(symbol: Symbol, meaning: SymbolFlags, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): EntityName | undefined;
  symbolToExpression(symbol: Symbol, meaning: SymbolFlags, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): Expression | undefined;
  symbolToTypeParameterDeclarations(symbol: Symbol, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): NodeArray<TypeParameterDeclaration> | undefined;
  symbolToParameterDeclaration(symbol: Symbol, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): ParameterDeclaration | undefined;
  typeParameterToDeclaration(parameter: TypeParameter, enclosingDeclaration: Node | undefined, flags: NodeBuilderFlags | undefined): TypeParameterDeclaration | undefined;

}
```

### Type Interface

Core type representation.

```typescript { .api }
interface Type {
  flags: TypeFlags;
  symbol: Symbol;
  pattern?: DestructuringPattern;
  aliasSymbol?: Symbol;
  aliasTypeArguments?: readonly Type[];
  
  getFlags(): TypeFlags;
  getSymbol(): Symbol | undefined;
  getProperties(): Symbol[];
  getProperty(propertyName: string): Symbol | undefined;
  getApparentProperties(): Symbol[];
  getCallSignatures(): readonly Signature[];
  getConstructSignatures(): readonly Signature[];
  getStringIndexType(): Type | undefined;
  getNumberIndexType(): Type | undefined;
  getBaseTypes(): BaseType[] | undefined;
  getNonNullableType(): Type;
  getConstraint(): Type | undefined;
  getDefault(): Type | undefined;
  isUnion(): this is UnionType;
  isIntersection(): this is IntersectionType;
  isUnionOrIntersection(): this is UnionOrIntersectionType;
  isLiteral(): this is LiteralType;
  isStringLiteral(): this is StringLiteralType;
  isNumberLiteral(): this is NumberLiteralType;
  isTypeParameter(): this is TypeParameter;
  isClassOrInterface(): this is InterfaceType;
  isClass(): this is InterfaceType;
}

enum TypeFlags {
  Any = 1,
  Unknown = 2,
  String = 4,
  Number = 8,
  Boolean = 16,
  Enum = 32,
  BigInt = 64,
  StringLiteral = 128,
  NumberLiteral = 256,
  BooleanLiteral = 512,
  EnumLiteral = 1024,
  BigIntLiteral = 2048,
  ESSymbol = 4096,
  UniqueESSymbol = 8192,
  Void = 16384,
  Undefined = 32768,
  Null = 65536,
  Never = 131072,
  TypeParameter = 262144,
  Object = 524288,
  Union = 1048576,
  Intersection = 2097152,
  Index = 4194304,
  IndexedAccess = 8388608,
  Conditional = 16777216,
  Substitution = 33554432,
  NonPrimitive = 67108864,
  TemplateLiteral = 134217728,
  StringMapping = 268435456,
  Literal = 2944,
  Unit = 109472,
  Freshable = 2976,
  StringOrNumberLiteral = 384,
  PossiblyFalsy = 117724,
  StringLike = 402653316,
  NumberLike = 296,
  BigIntLike = 2112,
  BooleanLike = 528,
  EnumLike = 1056,
  ESSymbolLike = 12288,
  VoidLike = 49152,
  UnionOrIntersection = 3145728,
  StructuredType = 3670016,
  TypeVariable = 8650752,
  InstantiableNonPrimitive = 58982400,
  InstantiablePrimitive = 406847488,
  Instantiable = 465829888,
  StructuredOrInstantiable = 469499904,
  Narrowable = 536624127
}
```

### Utility Types

Fundamental utility types used throughout the type system.

```typescript { .api }
/**
 * String type branded for use as an escaped identifier name.
 * Used internally for efficient Symbol name storage.
 */
type __String = (string & {
  __escapedIdentifier: void;
}) | (void & {
  __escapedIdentifier: void;
});

/**
 * String type branded as a file system path.
 */
type Path = string & {
  __pathBrand: any;
};

/**
 * Map of symbol names to Symbol objects.
 */
type SymbolTable = Map<__String, Symbol>;

/**
 * Generic wrapper that adds optional metadata to any type.
 */
type WithMetadata<T> = T & {
  metadata?: unknown;
};

/**
 * Generic string-indexed object interface.
 */
interface MapLike<T> {
  [index: string]: T;
}
```

### Symbol Interface

Represents a named declaration.

```typescript { .api }
interface Symbol {
  flags: SymbolFlags;
  escapedName: __String;
  readonly name: string;
  declarations?: Declaration[];
  valueDeclaration?: Declaration;
  members?: SymbolTable;
  exports?: SymbolTable;
  globalExports?: SymbolTable;

  getFlags(): SymbolFlags;
  getName(): string;
  getEscapedName(): __String;
  getDeclarations(): Declaration[] | undefined;
  getDocumentationComment(typeChecker: TypeChecker | undefined): SymbolDisplayPart[];
  getJsDocTags(checker?: TypeChecker): JSDocTagInfo[];
}

enum SymbolFlags {
  None = 0,
  FunctionScopedVariable = 1,
  BlockScopedVariable = 2,
  Property = 4,
  EnumMember = 8,
  Function = 16,
  Class = 32,
  Interface = 64,
  ConstEnum = 128,
  RegularEnum = 256,
  ValueModule = 512,
  NamespaceModule = 1024,
  TypeLiteral = 2048,
  ObjectLiteral = 4096,
  Method = 8192,
  Constructor = 16384,
  GetAccessor = 32768,
  SetAccessor = 65536,
  Signature = 131072,
  TypeParameter = 262144,
  TypeAlias = 524288,
  ExportValue = 1048576,
  Alias = 2097152,
  Prototype = 4194304,
  ExportStar = 8388608,
  Optional = 16777216,
  Transient = 33554432,
  Assignment = 67108864,
  ModuleExports = 134217728,
  All = -1,
  Enum = 384,
  Variable = 3,
  Value = 111551,
  Type = 788968,
  Namespace = 1920,
  Module = 1536,
  Accessor = 98304,
  FunctionScopedVariableExcludes = 111550,
  BlockScopedVariableExcludes = 111551,
  ParameterExcludes = 111551,
  PropertyExcludes = 0,
  EnumMemberExcludes = 900095,
  FunctionExcludes = 110991,
  ClassExcludes = 899503,
  InterfaceExcludes = 788872,
  RegularEnumExcludes = 899327,
  ConstEnumExcludes = 899967,
  ValueModuleExcludes = 110735,
  NamespaceModuleExcludes = 0,
  MethodExcludes = 103359,
  GetAccessorExcludes = 46015,
  SetAccessorExcludes = 78783,
  AccessorExcludes = 13247,
  TypeParameterExcludes = 526824,
  TypeAliasExcludes = 788968,
  AliasExcludes = 2097152,
  ModuleMember = 2623475,
  ExportHasLocal = 944,
  BlockScoped = 418,
  PropertyOrAccessor = 98308,
  ClassMember = 106500
}
```

### Signature Interface

Represents function, constructor, or index signatures.

```typescript { .api }
interface Signature {
  declaration?: SignatureDeclaration | JSDocSignature;
  typeParameters?: readonly TypeParameter[];
  parameters: readonly Symbol[];
  thisParameter?: Symbol;
  
  getDeclaration(): SignatureDeclaration;
  getTypeParameters(): TypeParameter[] | undefined;
  getParameters(): Symbol[];
  getTypeParameterAtPosition(pos: number): Type;
  getReturnType(): Type;
  getDocumentationComment(typeChecker: TypeChecker | undefined): SymbolDisplayPart[];
  getJsDocTags(): JSDocTagInfo[];
}

enum SignatureKind {
  Call = 0,
  Construct = 1
}
```

### Specialized Type Interfaces

```typescript { .api }
interface ObjectType extends Type {
  objectFlags: ObjectFlags;
}

interface InterfaceType extends ObjectType {
  typeParameters: TypeParameter[] | undefined;
  outerTypeParameters: TypeParameter[] | undefined;
  localTypeParameters: TypeParameter[] | undefined;
  thisType: TypeParameter | undefined;
  resolvedBaseConstructorType?: Type;
  resolvedBaseTypes: BaseType[];
}

interface TypeReference extends ObjectType {
  target: GenericType;
  node?: TypeReferenceNode | ArrayTypeNode | TupleTypeNode;
  resolvedTypeArguments?: readonly Type[];
  literalType?: TypeReference;
}

interface UnionOrIntersectionType extends Type {
  types: Type[];
  propertyCache?: SymbolTable;
  propertyCacheWithoutObjectFunctionPropertyAugment?: SymbolTable;
  resolvedProperties: Symbol[];
  resolvedIndexType: IndexType;
  resolvedStringIndexType: IndexType;
  resolvedBaseConstraint: Type;
}

interface UnionType extends UnionOrIntersectionType {
  resolvedReducedType?: Type;
}

interface IntersectionType extends UnionOrIntersectionType {
  resolvedApparentType: Type;
}

interface TypeParameter extends InstantiableType {
}

interface IndexedAccessType extends InstantiableType {
  objectType: Type;
  indexType: Type;
  constraint?: Type;
  simplifiedForReading?: Type;
  simplifiedForWriting?: Type;
}

interface ConditionalRoot {
  node: ConditionalTypeNode;
  checkType: Type;
  extendsType: Type;
  isDistributive: boolean;
  inferTypeParameters?: TypeParameter[];
  outerTypeParameters?: TypeParameter[];
  instantiations?: Map<string, Type>;
  aliasSymbol?: Symbol;
  aliasTypeArguments?: Type[];
}

interface ConditionalType extends InstantiableType {
  root: ConditionalRoot;
  checkType: Type;
  extendsType: Type;
  resolvedTrueType?: Type;
  resolvedFalseType?: Type;
}

interface SubstitutionType extends InstantiableType {
  baseType: Type;
  substitute: Type;
}

interface StringLiteralType extends LiteralType {
  value: string;
}

interface NumberLiteralType extends LiteralType {
  value: number;
}

interface BigIntLiteralType extends LiteralType {
  value: PseudoBigInt;
}

interface EnumType extends Type {
  memberTypes: Type[];
}

/**
 * Represents a BigInt value in the type system.
 * BigInt values are stored as strings to avoid precision issues.
 */
interface PseudoBigInt {
  /** Whether the BigInt is negative */
  negative: boolean;
  /** The base-10 string representation of the BigInt value */
  base10Value: string;
}

/**
 * Base type for interface types, intersection types, and type variables.
 * Used as the result type for getBaseTypes() methods.
 */
type BaseType = ObjectType | IntersectionType | TypeVariable;
```

### Index Information

```typescript { .api }
interface IndexInfo {
  keyType: Type;
  type: Type;
  isReadonly: boolean;
  declaration?: IndexSignatureDeclaration;
}

enum IndexKind {
  String = 0,
  Number = 1
}
```

### Type Predicates

```typescript { .api }
interface TypePredicate {
  kind: TypePredicateKind;
  type: Type | undefined;
}

interface TypePredicateBase {
  kind: TypePredicateKind;
  type: Type | undefined;
}

interface ThisTypePredicate extends TypePredicateBase {
  kind: TypePredicateKind.This;
  parameterName: undefined;
  parameterIndex: undefined;
  type: Type;
}

interface IdentifierTypePredicate extends TypePredicateBase {
  kind: TypePredicateKind.Identifier;
  parameterName: string;
  parameterIndex: number;
  type: Type;
}

interface AssertsThisTypePredicate extends TypePredicateBase {
  kind: TypePredicateKind.AssertsThis;
  parameterName: undefined;
  parameterIndex: undefined;
  type: Type | undefined;
}

interface AssertsIdentifierTypePredicate extends TypePredicateBase {
  kind: TypePredicateKind.AssertsIdentifier;
  parameterName: string;
  parameterIndex: number;
  type: Type | undefined;
}

enum TypePredicateKind {
  This = 0,
  Identifier = 1,
  AssertsThis = 2,
  AssertsIdentifier = 3
}
```

### Type Format Flags

```typescript { .api }
enum TypeFormatFlags {
  None = 0,
  NoTruncation = 1,
  WriteArrayAsGenericType = 2,
  UseStructuralFallback = 4,
  WriteTypeArgumentsOfSignature = 8,
  UseFullyQualifiedType = 16,
  SuppressAnyReturnType = 32,
  MultilineObjectLiterals = 64,
  WriteClassExpressionAsTypeLiteral = 128,
  UseTypeOfFunction = 256,
  OmitParameterModifiers = 512,
  UseAliasDefinedOutsideCurrentScope = 1024,
  UseSingleQuotesForStringLiteralType = 2048,
  NoTypeReduction = 4096,
  OmitThisParameter = 8192,
  AllowUniqueESSymbolType = 16384,
  AddUndefined = 32768,
  WriteArrowStyleSignature = 65536,
  InArrayType = 131072,
  InElementType = 262144,
  InFirstTypeArgument = 524288,
  InTypeAlias = 1048576,
  WriteOwnNameForAnyLike = 0,
  NodeBuilderFlagsMask = 848330091
}

enum SymbolFormatFlags {
  None = 0,
  WriteTypeParametersOrArguments = 1,
  UseOnlyExternalAliasing = 2,
  AllowAnyNodeKind = 4,
  UseAliasDefinedOutsideCurrentScope = 8,
  DoNotIncludeSymbolChain = 16,
  IncludeInstanceAndStaticMembers = 32
}
```

### Usage Example

Analyzing types with TypeChecker:

```typescript
import * as ts from 'typescript';

const code = `
interface User {
  name: string;
  age: number;
}

const user: User = { name: 'Alice', age: 30 };
`;

const sourceFile = ts.createSourceFile('example.ts', code, ts.ScriptTarget.Latest, true);
const program = ts.createProgram(['example.ts'], {}, {
  getSourceFile: (fileName) => fileName === 'example.ts' ? sourceFile : undefined,
  writeFile: () => {},
  getCurrentDirectory: () => '',
  getDirectories: () => [],
  fileExists: () => true,
  readFile: () => '',
  getCanonicalFileName: (fileName) => fileName,
  useCaseSensitiveFileNames: () => true,
  getNewLine: () => '\n'
});

const checker = program.getTypeChecker();

// Visit nodes and analyze types
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText() === 'user') {
    const type = checker.getTypeAtLocation(node);
    console.log('Type:', checker.typeToString(type));
    
    const properties = type.getProperties();
    properties.forEach(prop => {
      const propType = checker.getTypeOfSymbolAtLocation(prop, node);
      console.log(`  ${prop.getName()}: ${checker.typeToString(propType)}`);
    });
  }
  
  ts.forEachChild(node, visit);
}

visit(sourceFile);
```
