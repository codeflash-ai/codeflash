# Types and Symbols

Type and Symbol interfaces for representing TypeScript types and declarations.

## Type Interface

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

## Symbol Interface

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

## Signature Interface

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

## Specialized Type Interfaces

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

interface PseudoBigInt {
  negative: boolean;
  base10Value: string;
}

type BaseType = ObjectType | IntersectionType | TypeVariable;
```

## Index Information

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

## Type Predicates

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

## Type Format Flags

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

## Utility Types

Fundamental utility types used throughout the type system.

```typescript { .api }
type __String = (string & {
  __escapedIdentifier: void;
}) | (void & {
  __escapedIdentifier: void;
});

type Path = string & {
  __pathBrand: any;
};

type SymbolTable = Map<__String, Symbol>;

type WithMetadata<T> = T & {
  metadata?: unknown;
};

interface MapLike<T> {
  [index: string]: T;
}
```

