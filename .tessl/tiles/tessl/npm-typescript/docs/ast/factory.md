# Node Factory

Factory functions for creating and updating AST nodes programmatically. The `factory` constant provides 300+ methods for building complete TypeScript AST structures.

## Capabilities

### Factory Constant

```typescript { .api }
const factory: NodeFactory;
```

### Node Factory Interface

Core factory methods for AST node creation.

```typescript { .api }
interface NodeFactory {
  /* Literals */
  createNumericLiteral(value: string | number, numericLiteralFlags?: TokenFlags): NumericLiteral;
  createBigIntLiteral(value: string | PseudoBigInt): BigIntLiteral;
  createStringLiteral(text: string, isSingleQuote?: boolean): StringLiteral;
  createStringLiteralFromNode(sourceNode: PropertyNameLiteral | PrivateIdentifier, isSingleQuote?: boolean): StringLiteral;
  createRegularExpressionLiteral(text: string): RegularExpressionLiteral;
  
  /* Identifiers */
  createIdentifier(text: string): Identifier;
  createTempVariable(recordTempVariable: ((node: Identifier) => void) | undefined, reservedInNestedScopes?: boolean): Identifier;
  createLoopVariable(reservedInNestedScopes?: boolean): Identifier;
  createUniqueName(text: string, flags?: GeneratedIdentifierFlags): Identifier;
  getGeneratedNameForNode(node: Node | undefined, flags?: GeneratedIdentifierFlags): Identifier;
  createPrivateIdentifier(text: string): PrivateIdentifier;
  createUniquePrivateName(text?: string): PrivateIdentifier;
  
  /* Punctuation */
  createToken<TKind extends PunctuationSyntaxKind>(token: TKind): PunctuationToken<TKind>;
  createSuper(): SuperExpression;
  createThis(): ThisExpression;
  createNull(): NullLiteral;
  createTrue(): TrueLiteral;
  createFalse(): FalseLiteral;
  
  /* Modifiers */
  createModifier<T extends ModifierSyntaxKind>(kind: T): ModifierToken<T>;
  createModifiersFromModifierFlags(flags: ModifierFlags): Modifier[] | undefined;
  
  /* QualifiedName */
  createQualifiedName(left: EntityName, right: string | Identifier): QualifiedName;
  updateQualifiedName(node: QualifiedName, left: EntityName, right: Identifier): QualifiedName;
  
  /* Computed Property Name */
  createComputedPropertyName(expression: Expression): ComputedPropertyName;
  updateComputedPropertyName(node: ComputedPropertyName, expression: Expression): ComputedPropertyName;
  
  /* Type Parameters */
  createTypeParameterDeclaration(modifiers: readonly Modifier[] | undefined, name: string | Identifier, constraint?: TypeNode, defaultType?: TypeNode): TypeParameterDeclaration;
  updateTypeParameterDeclaration(node: TypeParameterDeclaration, modifiers: readonly Modifier[] | undefined, name: Identifier, constraint: TypeNode | undefined, defaultType: TypeNode | undefined): TypeParameterDeclaration;
  
  /* Parameters */
  createParameterDeclaration(modifiers: readonly ModifierLike[] | undefined, dotDotDotToken: DotDotDotToken | undefined, name: string | BindingName, questionToken?: QuestionToken, type?: TypeNode, initializer?: Expression): ParameterDeclaration;
  updateParameterDeclaration(node: ParameterDeclaration, modifiers: readonly ModifierLike[] | undefined, dotDotDotToken: DotDotDotToken | undefined, name: string | BindingName, questionToken: QuestionToken | undefined, type: TypeNode | undefined, initializer: Expression | undefined): ParameterDeclaration;
  
  /* Decorators */
  createDecorator(expression: Expression): Decorator;
  updateDecorator(node: Decorator, expression: Expression): Decorator;
  
  /* Type Nodes */
  createKeywordTypeNode<TKind extends KeywordTypeSyntaxKind>(kind: TKind): KeywordTypeNode<TKind>;

  createTypeReferenceNode(typeName: string | EntityName, typeArguments?: readonly TypeNode[]): TypeReferenceNode;
  updateTypeReferenceNode(node: TypeReferenceNode, typeName: EntityName, typeArguments: NodeArray<TypeNode> | undefined): TypeReferenceNode;
  
  createFunctionTypeNode(typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode): FunctionTypeNode;
  updateFunctionTypeNode(node: FunctionTypeNode, typeParameters: NodeArray<TypeParameterDeclaration> | undefined, parameters: NodeArray<ParameterDeclaration>, type: TypeNode): FunctionTypeNode;
  
  createConstructorTypeNode(modifiers: readonly Modifier[] | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode): ConstructorTypeNode;
  updateConstructorTypeNode(node: ConstructorTypeNode, modifiers: readonly Modifier[] | undefined, typeParameters: NodeArray<TypeParameterDeclaration> | undefined, parameters: NodeArray<ParameterDeclaration>, type: TypeNode): ConstructorTypeNode;
  
  createTypeQueryNode(exprName: EntityName, typeArguments?: readonly TypeNode[]): TypeQueryNode;
  updateTypeQueryNode(node: TypeQueryNode, exprName: EntityName, typeArguments?: readonly TypeNode[]): TypeQueryNode;
  
  createTypeLiteralNode(members: readonly TypeElement[] | undefined): TypeLiteralNode;
  updateTypeLiteralNode(node: TypeLiteralNode, members: NodeArray<TypeElement>): TypeLiteralNode;
  
  createArrayTypeNode(elementType: TypeNode): ArrayTypeNode;
  updateArrayTypeNode(node: ArrayTypeNode, elementType: TypeNode): ArrayTypeNode;
  
  createTupleTypeNode(elements: readonly (TypeNode | NamedTupleMember)[]): TupleTypeNode;
  updateTupleTypeNode(node: TupleTypeNode, elements: readonly (TypeNode | NamedTupleMember)[]): TupleTypeNode;
  
  createUnionTypeNode(types: readonly TypeNode[]): UnionTypeNode;
  updateUnionTypeNode(node: UnionTypeNode, types: NodeArray<TypeNode>): UnionTypeNode;
  
  createIntersectionTypeNode(types: readonly TypeNode[]): IntersectionTypeNode;
  updateIntersectionTypeNode(node: IntersectionTypeNode, types: NodeArray<TypeNode>): IntersectionTypeNode;
  
  createConditionalTypeNode(checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode): ConditionalTypeNode;
  updateConditionalTypeNode(node: ConditionalTypeNode, checkType: TypeNode, extendsType: TypeNode, trueType: TypeNode, falseType: TypeNode): ConditionalTypeNode;
  
  createInferTypeNode(typeParameter: TypeParameterDeclaration): InferTypeNode;
  updateInferTypeNode(node: InferTypeNode, typeParameter: TypeParameterDeclaration): InferTypeNode;
  
  createMappedTypeNode(readonlyToken: ReadonlyKeyword | PlusToken | MinusToken | undefined, typeParameter: TypeParameterDeclaration, nameType: TypeNode | undefined, questionToken: QuestionToken | PlusToken | MinusToken | undefined, type: TypeNode | undefined, members: NodeArray<TypeElement> | undefined): MappedTypeNode;
  updateMappedTypeNode(node: MappedTypeNode, readonlyToken: ReadonlyKeyword | PlusToken | MinusToken | undefined, typeParameter: TypeParameterDeclaration, nameType: TypeNode | undefined, questionToken: QuestionToken | PlusToken | MinusToken | undefined, type: TypeNode | undefined, members: NodeArray<TypeElement> | undefined): MappedTypeNode;
  
  createLiteralTypeNode(literal: LiteralTypeNode["literal"]): LiteralTypeNode;
  updateLiteralTypeNode(node: LiteralTypeNode, literal: LiteralTypeNode["literal"]): LiteralTypeNode;
  
  /* Expressions */
  createObjectLiteralExpression(properties?: readonly ObjectLiteralElementLike[], multiLine?: boolean): ObjectLiteralExpression;
  updateObjectLiteralExpression(node: ObjectLiteralExpression, properties: readonly ObjectLiteralElementLike[]): ObjectLiteralExpression;
  
  createPropertyAccessExpression(expression: Expression, name: string | Identifier | PrivateIdentifier): PropertyAccessExpression;
  updatePropertyAccessExpression(node: PropertyAccessExpression, expression: Expression, name: Identifier | PrivateIdentifier): PropertyAccessExpression;
  
  createPropertyAccessChain(expression: Expression, questionDotToken: QuestionDotToken | undefined, name: string | Identifier | PrivateIdentifier): PropertyAccessChain;
  updatePropertyAccessChain(node: PropertyAccessChain, expression: Expression, questionDotToken: QuestionDotToken | undefined, name: Identifier | PrivateIdentifier): PropertyAccessChain;
  
  createElementAccessExpression(expression: Expression, index: number | Expression): ElementAccessExpression;
  updateElementAccessExpression(node: ElementAccessExpression, expression: Expression, argumentExpression: Expression): ElementAccessExpression;
  
  createCallExpression(expression: Expression, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[] | undefined): CallExpression;
  updateCallExpression(node: CallExpression, expression: Expression, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[]): CallExpression;
  
  createNewExpression(expression: Expression, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[] | undefined): NewExpression;
  updateNewExpression(node: NewExpression, expression: Expression, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[] | undefined): NewExpression;
  
  createTaggedTemplateExpression(tag: Expression, typeArguments: readonly TypeNode[] | undefined, template: TemplateLiteral): TaggedTemplateExpression;
  updateTaggedTemplateExpression(node: TaggedTemplateExpression, tag: Expression, typeArguments: readonly TypeNode[] | undefined, template: TemplateLiteral): TaggedTemplateExpression;
  
  createTypeAssertion(type: TypeNode, expression: Expression): TypeAssertion;
  updateTypeAssertion(node: TypeAssertion, type: TypeNode, expression: Expression): TypeAssertion;
  
  createParenthesizedExpression(expression: Expression): ParenthesizedExpression;
  updateParenthesizedExpression(node: ParenthesizedExpression, expression: Expression): ParenthesizedExpression;
  
  createFunctionExpression(modifiers: readonly Modifier[] | undefined, asteriskToken: AsteriskToken | undefined, name: string | Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[] | undefined, type: TypeNode | undefined, body: Block): FunctionExpression;
  updateFunctionExpression(node: FunctionExpression, modifiers: readonly Modifier[] | undefined, asteriskToken: AsteriskToken | undefined, name: Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block): FunctionExpression;
  
  createArrowFunction(modifiers: readonly Modifier[] | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, equalsGreaterThanToken: EqualsGreaterThanToken | undefined, body: ConciseBody): ArrowFunction;
  updateArrowFunction(node: ArrowFunction, modifiers: readonly Modifier[] | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, equalsGreaterThanToken: EqualsGreaterThanToken, body: ConciseBody): ArrowFunction;
  
  createDeleteExpression(expression: Expression): DeleteExpression;
  updateDeleteExpression(node: DeleteExpression, expression: Expression): DeleteExpression;
  
  createTypeOfExpression(expression: Expression): TypeOfExpression;
  updateTypeOfExpression(node: TypeOfExpression, expression: Expression): TypeOfExpression;
  
  createVoidExpression(expression: Expression): VoidExpression;
  updateVoidExpression(node: VoidExpression, expression: Expression): VoidExpression;
  
  createAwaitExpression(expression: Expression): AwaitExpression;
  updateAwaitExpression(node: AwaitExpression, expression: Expression): AwaitExpression;
  
  createPrefixUnaryExpression(operator: PrefixUnaryOperator, operand: Expression): PrefixUnaryExpression;
  updatePrefixUnaryExpression(node: PrefixUnaryExpression, operand: Expression): PrefixUnaryExpression;
  
  createPostfixUnaryExpression(operand: Expression, operator: PostfixUnaryOperator): PostfixUnaryExpression;
  updatePostfixUnaryExpression(node: PostfixUnaryExpression, operand: Expression): PostfixUnaryExpression;
  
  createBinaryExpression(left: Expression, operator: BinaryOperator | BinaryOperatorToken, right: Expression): BinaryExpression;
  updateBinaryExpression(node: BinaryExpression, left: Expression, operator: BinaryOperatorToken, right: Expression): BinaryExpression;
  
  createConditionalExpression(condition: Expression, questionToken: QuestionToken | undefined, whenTrue: Expression, colonToken: ColonToken | undefined, whenFalse: Expression): ConditionalExpression;
  updateConditionalExpression(node: ConditionalExpression, condition: Expression, questionToken: QuestionToken, whenTrue: Expression, colonToken: ColonToken, whenFalse: Expression): ConditionalExpression;
  
  createArrayLiteralExpression(elements?: readonly Expression[], multiLine?: boolean): ArrayLiteralExpression;
  updateArrayLiteralExpression(node: ArrayLiteralExpression, elements: readonly Expression[]): ArrayLiteralExpression;
  
  createSpreadElement(expression: Expression): SpreadElement;
  updateSpreadElement(node: SpreadElement, expression: Expression): SpreadElement;
  
  createClassExpression(modifiers: readonly ModifierLike[] | undefined, name: string | Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly ClassElement[]): ClassExpression;
  updateClassExpression(node: ClassExpression, modifiers: readonly ModifierLike[] | undefined, name: Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly ClassElement[]): ClassExpression;
  
  createAsExpression(expression: Expression, type: TypeNode): AsExpression;
  updateAsExpression(node: AsExpression, expression: Expression, type: TypeNode): AsExpression;
  
  createNonNullExpression(expression: Expression): NonNullExpression;
  updateNonNullExpression(node: NonNullExpression, expression: Expression): NonNullExpression;
  
  createSatisfiesExpression(expression: Expression, type: TypeNode): SatisfiesExpression;
  updateSatisfiesExpression(node: SatisfiesExpression, expression: Expression, type: TypeNode): SatisfiesExpression;
  
  /* Statements */
  createBlock(statements: readonly Statement[], multiLine?: boolean): Block;
  updateBlock(node: Block, statements: readonly Statement[]): Block;
  
  createVariableStatement(modifiers: readonly Modifier[] | undefined, declarationList: VariableDeclarationList | readonly VariableDeclaration[]): VariableStatement;
  updateVariableStatement(node: VariableStatement, modifiers: readonly Modifier[] | undefined, declarationList: VariableDeclarationList): VariableStatement;
  
  createEmptyStatement(): EmptyStatement;
  
  createExpressionStatement(expression: Expression): ExpressionStatement;
  updateExpressionStatement(node: ExpressionStatement, expression: Expression): ExpressionStatement;
  
  createIfStatement(expression: Expression, thenStatement: Statement, elseStatement?: Statement): IfStatement;
  updateIfStatement(node: IfStatement, expression: Expression, thenStatement: Statement, elseStatement: Statement | undefined): IfStatement;
  
  createDoStatement(statement: Statement, expression: Expression): DoStatement;
  updateDoStatement(node: DoStatement, statement: Statement, expression: Expression): DoStatement;
  
  createWhileStatement(expression: Expression, statement: Statement): WhileStatement;
  updateWhileStatement(node: WhileStatement, expression: Expression, statement: Statement): WhileStatement;
  
  createForStatement(initializer: ForInitializer | undefined, condition: Expression | undefined, incrementor: Expression | undefined, statement: Statement): ForStatement;
  updateForStatement(node: ForStatement, initializer: ForInitializer | undefined, condition: Expression | undefined, incrementor: Expression | undefined, statement: Statement): ForStatement;
  
  createForInStatement(initializer: ForInitializer, expression: Expression, statement: Statement): ForInStatement;
  updateForInStatement(node: ForInStatement, initializer: ForInitializer, expression: Expression, statement: Statement): ForInStatement;
  
  createForOfStatement(awaitModifier: AwaitKeyword | undefined, initializer: ForInitializer, expression: Expression, statement: Statement): ForOfStatement;
  updateForOfStatement(node: ForOfStatement, awaitModifier: AwaitKeyword | undefined, initializer: ForInitializer, expression: Expression, statement: Statement): ForOfStatement;
  
  createContinueStatement(label?: string | Identifier): ContinueStatement;
  updateContinueStatement(node: ContinueStatement, label: Identifier | undefined): ContinueStatement;
  
  createBreakStatement(label?: string | Identifier): BreakStatement;
  updateBreakStatement(node: BreakStatement, label: Identifier | undefined): BreakStatement;
  
  createReturnStatement(expression?: Expression): ReturnStatement;
  updateReturnStatement(node: ReturnStatement, expression: Expression | undefined): ReturnStatement;
  
  createSwitchStatement(expression: Expression, caseBlock: CaseBlock): SwitchStatement;
  updateSwitchStatement(node: SwitchStatement, expression: Expression, caseBlock: CaseBlock): SwitchStatement;
  
  createThrowStatement(expression: Expression): ThrowStatement;
  updateThrowStatement(node: ThrowStatement, expression: Expression): ThrowStatement;
  
  createTryStatement(tryBlock: Block, catchClause: CatchClause | undefined, finallyBlock: Block | undefined): TryStatement;
  updateTryStatement(node: TryStatement, tryBlock: Block, catchClause: CatchClause | undefined, finallyBlock: Block | undefined): TryStatement;
  
  /* Declarations */
  createVariableDeclaration(name: string | BindingName, exclamationToken?: ExclamationToken, type?: TypeNode, initializer?: Expression): VariableDeclaration;
  updateVariableDeclaration(node: VariableDeclaration, name: BindingName, exclamationToken: ExclamationToken | undefined, type: TypeNode | undefined, initializer: Expression | undefined): VariableDeclaration;
  
  createVariableDeclarationList(declarations: readonly VariableDeclaration[], flags?: NodeFlags): VariableDeclarationList;
  updateVariableDeclarationList(node: VariableDeclarationList, declarations: readonly VariableDeclaration[]): VariableDeclarationList;
  
  createFunctionDeclaration(modifiers: readonly ModifierLike[] | undefined, asteriskToken: AsteriskToken | undefined, name: string | Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): FunctionDeclaration;
  updateFunctionDeclaration(node: FunctionDeclaration, modifiers: readonly ModifierLike[] | undefined, asteriskToken: AsteriskToken | undefined, name: Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): FunctionDeclaration;
  
  createClassDeclaration(modifiers: readonly ModifierLike[] | undefined, name: string | Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly ClassElement[]): ClassDeclaration;
  updateClassDeclaration(node: ClassDeclaration, modifiers: readonly ModifierLike[] | undefined, name: Identifier | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly ClassElement[]): ClassDeclaration;

  createPropertyDeclaration(modifiers: readonly ModifierLike[] | undefined, name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken | undefined, type: TypeNode | undefined, initializer: Expression | undefined): PropertyDeclaration;
  updatePropertyDeclaration(node: PropertyDeclaration, modifiers: readonly ModifierLike[] | undefined, name: string | PropertyName, questionOrExclamationToken: QuestionToken | ExclamationToken | undefined, type: TypeNode | undefined, initializer: Expression | undefined): PropertyDeclaration;

  createMethodDeclaration(modifiers: readonly ModifierLike[] | undefined, asteriskToken: AsteriskToken | undefined, name: string | PropertyName, questionToken: QuestionToken | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): MethodDeclaration;
  updateMethodDeclaration(node: MethodDeclaration, modifiers: readonly ModifierLike[] | undefined, asteriskToken: AsteriskToken | undefined, name: PropertyName, questionToken: QuestionToken | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): MethodDeclaration;

  createConstructorDeclaration(modifiers: readonly ModifierLike[] | undefined, parameters: readonly ParameterDeclaration[], body: Block | undefined): ConstructorDeclaration;
  updateConstructorDeclaration(node: ConstructorDeclaration, modifiers: readonly ModifierLike[] | undefined, parameters: readonly ParameterDeclaration[], body: Block | undefined): ConstructorDeclaration;

  createInterfaceDeclaration(modifiers: readonly Modifier[] | undefined, name: string | Identifier, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly TypeElement[]): InterfaceDeclaration;
  updateInterfaceDeclaration(node: InterfaceDeclaration, modifiers: readonly Modifier[] | undefined, name: Identifier, typeParameters: readonly TypeParameterDeclaration[] | undefined, heritageClauses: readonly HeritageClause[] | undefined, members: readonly TypeElement[]): InterfaceDeclaration;
  
  createTypeAliasDeclaration(modifiers: readonly Modifier[] | undefined, name: string | Identifier, typeParameters: readonly TypeParameterDeclaration[] | undefined, type: TypeNode): TypeAliasDeclaration;
  updateTypeAliasDeclaration(node: TypeAliasDeclaration, modifiers: readonly Modifier[] | undefined, name: Identifier, typeParameters: readonly TypeParameterDeclaration[] | undefined, type: TypeNode): TypeAliasDeclaration;
  
  createEnumDeclaration(modifiers: readonly Modifier[] | undefined, name: string | Identifier, members: readonly EnumMember[]): EnumDeclaration;
  updateEnumDeclaration(node: EnumDeclaration, modifiers: readonly Modifier[] | undefined, name: Identifier, members: readonly EnumMember[]): EnumDeclaration;
  
  createModuleDeclaration(modifiers: readonly Modifier[] | undefined, name: ModuleName, body: ModuleBody | undefined, flags?: NodeFlags): ModuleDeclaration;
  updateModuleDeclaration(node: ModuleDeclaration, modifiers: readonly Modifier[] | undefined, name: ModuleName, body: ModuleBody | undefined): ModuleDeclaration;
  
  createImportEqualsDeclaration(modifiers: readonly Modifier[] | undefined, isTypeOnly: boolean, name: string | Identifier, moduleReference: ModuleReference): ImportEqualsDeclaration;
  updateImportEqualsDeclaration(node: ImportEqualsDeclaration, modifiers: readonly Modifier[] | undefined, isTypeOnly: boolean, name: Identifier, moduleReference: ModuleReference): ImportEqualsDeclaration;
  
  createImportDeclaration(modifiers: readonly Modifier[] | undefined, importClause: ImportClause | undefined, moduleSpecifier: Expression, attributes?: ImportAttributes): ImportDeclaration;
  updateImportDeclaration(node: ImportDeclaration, modifiers: readonly Modifier[] | undefined, importClause: ImportClause | undefined, moduleSpecifier: Expression, attributes: ImportAttributes | undefined): ImportDeclaration;
  
  createExportAssignment(modifiers: readonly Modifier[] | undefined, isExportEquals: boolean | undefined, expression: Expression): ExportAssignment;
  updateExportAssignment(node: ExportAssignment, modifiers: readonly Modifier[] | undefined, expression: Expression): ExportAssignment;
  
  createExportDeclaration(modifiers: readonly Modifier[] | undefined, isTypeOnly: boolean, exportClause: NamedExportBindings | undefined, moduleSpecifier?: Expression, attributes?: ImportAttributes): ExportDeclaration;
  updateExportDeclaration(node: ExportDeclaration, modifiers: readonly Modifier[] | undefined, isTypeOnly: boolean, exportClause: NamedExportBindings | undefined, moduleSpecifier: Expression | undefined, attributes: ImportAttributes | undefined): ExportDeclaration;
  
  /* Source File */
  createSourceFile(statements: readonly Statement[], endOfFileToken: EndOfFileToken, flags: NodeFlags): SourceFile;
  updateSourceFile(node: SourceFile, statements: readonly Statement[], isDeclarationFile?: boolean, referencedFiles?: readonly FileReference[], typeReferences?: readonly FileReference[], hasNoDefaultLib?: boolean, libReferences?: readonly FileReference[]): SourceFile;

  /* Type Signatures */
  createPropertySignature(modifiers: readonly Modifier[] | undefined, name: PropertyName | string, questionToken: QuestionToken | undefined, type: TypeNode | undefined): PropertySignature;
  updatePropertySignature(node: PropertySignature, modifiers: readonly Modifier[] | undefined, name: PropertyName, questionToken: QuestionToken | undefined, type: TypeNode | undefined): PropertySignature;
  createMethodSignature(modifiers: readonly Modifier[] | undefined, name: string | PropertyName, questionToken: QuestionToken | undefined, typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined): MethodSignature;
  updateMethodSignature(node: MethodSignature, modifiers: readonly Modifier[] | undefined, name: PropertyName, questionToken: QuestionToken | undefined, typeParameters: NodeArray<TypeParameterDeclaration> | undefined, parameters: NodeArray<ParameterDeclaration>, type: TypeNode | undefined): MethodSignature;
  createCallSignature(typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined): CallSignatureDeclaration;
  updateCallSignature(node: CallSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration> | undefined, parameters: NodeArray<ParameterDeclaration>, type: TypeNode | undefined): CallSignatureDeclaration;
  createConstructSignature(typeParameters: readonly TypeParameterDeclaration[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined): ConstructSignatureDeclaration;
  updateConstructSignature(node: ConstructSignatureDeclaration, typeParameters: NodeArray<TypeParameterDeclaration> | undefined, parameters: NodeArray<ParameterDeclaration>, type: TypeNode | undefined): ConstructSignatureDeclaration;
  createIndexSignature(modifiers: readonly ModifierLike[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode): IndexSignatureDeclaration;
  updateIndexSignature(node: IndexSignatureDeclaration, modifiers: readonly ModifierLike[] | undefined, parameters: readonly ParameterDeclaration[], type: TypeNode): IndexSignatureDeclaration;
  createTemplateLiteralTypeSpan(type: TypeNode, literal: TemplateMiddle | TemplateTail): TemplateLiteralTypeSpan;
  updateTemplateLiteralTypeSpan(node: TemplateLiteralTypeSpan, type: TypeNode, literal: TemplateMiddle | TemplateTail): TemplateLiteralTypeSpan;

  /* Accessors & Class Members */
  createGetAccessorDeclaration(modifiers: readonly ModifierLike[] | undefined, name: string | PropertyName, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): GetAccessorDeclaration;
  updateGetAccessorDeclaration(node: GetAccessorDeclaration, modifiers: readonly ModifierLike[] | undefined, name: PropertyName, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined, body: Block | undefined): GetAccessorDeclaration;
  createSetAccessorDeclaration(modifiers: readonly ModifierLike[] | undefined, name: string | PropertyName, parameters: readonly ParameterDeclaration[], body: Block | undefined): SetAccessorDeclaration;
  updateSetAccessorDeclaration(node: SetAccessorDeclaration, modifiers: readonly ModifierLike[] | undefined, name: PropertyName, parameters: readonly ParameterDeclaration[], body: Block | undefined): SetAccessorDeclaration;
  createClassStaticBlockDeclaration(body: Block): ClassStaticBlockDeclaration;
  updateClassStaticBlockDeclaration(node: ClassStaticBlockDeclaration, body: Block): ClassStaticBlockDeclaration;
  createSemicolonClassElement(): SemicolonClassElement;

  /* Type Nodes */
  createTypePredicateNode(assertsModifier: AssertsKeyword | undefined, parameterName: Identifier | ThisTypeNode | string, type: TypeNode | undefined): TypePredicateNode;
  updateTypePredicateNode(node: TypePredicateNode, assertsModifier: AssertsKeyword | undefined, parameterName: Identifier | ThisTypeNode, type: TypeNode | undefined): TypePredicateNode;
  createNamedTupleMember(dotDotDotToken: DotDotDotToken | undefined, name: Identifier, questionToken: QuestionToken | undefined, type: TypeNode): NamedTupleMember;
  updateNamedTupleMember(node: NamedTupleMember, dotDotDotToken: DotDotDotToken | undefined, name: Identifier, questionToken: QuestionToken | undefined, type: TypeNode): NamedTupleMember;
  createOptionalTypeNode(type: TypeNode): OptionalTypeNode;
  updateOptionalTypeNode(node: OptionalTypeNode, type: TypeNode): OptionalTypeNode;
  createRestTypeNode(type: TypeNode): RestTypeNode;
  updateRestTypeNode(node: RestTypeNode, type: TypeNode): RestTypeNode;
  createImportTypeNode(argument: TypeNode, attributes?: ImportAttributes, qualifier?: EntityName, typeArguments?: readonly TypeNode[], isTypeOf?: boolean): ImportTypeNode;
  updateImportTypeNode(node: ImportTypeNode, argument: TypeNode, attributes: ImportAttributes | undefined, qualifier: EntityName | undefined, typeArguments: readonly TypeNode[] | undefined, isTypeOf?: boolean): ImportTypeNode;
  createParenthesizedType(type: TypeNode): ParenthesizedTypeNode;
  updateParenthesizedType(node: ParenthesizedTypeNode, type: TypeNode): ParenthesizedTypeNode;
  createThisTypeNode(): ThisTypeNode;
  createTypeOperatorNode(operator: SyntaxKind.KeyOfKeyword | SyntaxKind.UniqueKeyword | SyntaxKind.ReadonlyKeyword, type: TypeNode): TypeOperatorNode;
  updateTypeOperatorNode(node: TypeOperatorNode, type: TypeNode): TypeOperatorNode;
  createIndexedAccessTypeNode(objectType: TypeNode, indexType: TypeNode): IndexedAccessTypeNode;
  updateIndexedAccessTypeNode(node: IndexedAccessTypeNode, objectType: TypeNode, indexType: TypeNode): IndexedAccessTypeNode;
  createTemplateLiteralType(head: TemplateHead, templateSpans: readonly TemplateLiteralTypeSpan[]): TemplateLiteralTypeNode;
  updateTemplateLiteralType(node: TemplateLiteralTypeNode, head: TemplateHead, templateSpans: readonly TemplateLiteralTypeSpan[]): TemplateLiteralTypeNode;

  /* Binding Patterns */
  createObjectBindingPattern(elements: readonly BindingElement[]): ObjectBindingPattern;
  updateObjectBindingPattern(node: ObjectBindingPattern, elements: readonly BindingElement[]): ObjectBindingPattern;
  createArrayBindingPattern(elements: readonly ArrayBindingElement[]): ArrayBindingPattern;
  updateArrayBindingPattern(node: ArrayBindingPattern, elements: readonly ArrayBindingElement[]): ArrayBindingPattern;
  createBindingElement(dotDotDotToken: DotDotDotToken | undefined, propertyName: string | PropertyName | undefined, name: string | BindingName, initializer?: Expression): BindingElement;
  updateBindingElement(node: BindingElement, dotDotDotToken: DotDotDotToken | undefined, propertyName: PropertyName | undefined, name: BindingName, initializer: Expression | undefined): BindingElement;

  /* Statements */
  createWithStatement(expression: Expression, statement: Statement): WithStatement;
  updateWithStatement(node: WithStatement, expression: Expression, statement: Statement): WithStatement;
  createLabeledStatement(label: string | Identifier, statement: Statement): LabeledStatement;
  updateLabeledStatement(node: LabeledStatement, label: Identifier, statement: Statement): LabeledStatement;
  createDebuggerStatement(): DebuggerStatement;
  createNotEmittedStatement(original: Node): NotEmittedStatement;
  createNotEmittedTypeElement(): NotEmittedTypeElement;

  /* Expressions - Templates */
  createTemplateExpression(head: TemplateHead, templateSpans: readonly TemplateSpan[]): TemplateExpression;
  updateTemplateExpression(node: TemplateExpression, head: TemplateHead, templateSpans: readonly TemplateSpan[]): TemplateExpression;
  createTemplateHead(text: string, rawText?: string, templateFlags?: TokenFlags): TemplateHead;
  createTemplateHead(text: string | undefined, rawText: string, templateFlags?: TokenFlags): TemplateHead;
  createTemplateMiddle(text: string, rawText?: string, templateFlags?: TokenFlags): TemplateMiddle;
  createTemplateMiddle(text: string | undefined, rawText: string, templateFlags?: TokenFlags): TemplateMiddle;
  createTemplateTail(text: string, rawText?: string, templateFlags?: TokenFlags): TemplateTail;
  createTemplateTail(text: string | undefined, rawText: string, templateFlags?: TokenFlags): TemplateTail;
  createNoSubstitutionTemplateLiteral(text: string, rawText?: string): NoSubstitutionTemplateLiteral;
  createNoSubstitutionTemplateLiteral(text: string | undefined, rawText: string): NoSubstitutionTemplateLiteral;
  createTemplateSpan(expression: Expression, literal: TemplateMiddle | TemplateTail): TemplateSpan;
  updateTemplateSpan(node: TemplateSpan, expression: Expression, literal: TemplateMiddle | TemplateTail): TemplateSpan;

  /* Expressions - Yield & Other */
  createYieldExpression(asteriskToken: AsteriskToken, expression: Expression): YieldExpression;
  createYieldExpression(asteriskToken: undefined, expression: Expression | undefined): YieldExpression;
  updateYieldExpression(node: YieldExpression, asteriskToken: AsteriskToken | undefined, expression: Expression | undefined): YieldExpression;
  createOmittedExpression(): OmittedExpression;
  createExpressionWithTypeArguments(expression: Expression, typeArguments: readonly TypeNode[] | undefined): ExpressionWithTypeArguments;
  updateExpressionWithTypeArguments(node: ExpressionWithTypeArguments, expression: Expression, typeArguments: readonly TypeNode[] | undefined): ExpressionWithTypeArguments;
  createMetaProperty(keywordToken: MetaProperty["keywordToken"], name: Identifier): MetaProperty;
  updateMetaProperty(node: MetaProperty, name: Identifier): MetaProperty;
  createPartiallyEmittedExpression(expression: Expression, original?: Node): PartiallyEmittedExpression;
  updatePartiallyEmittedExpression(node: PartiallyEmittedExpression, expression: Expression): PartiallyEmittedExpression;
  createCommaListExpression(elements: readonly Expression[]): CommaListExpression;
  updateCommaListExpression(node: CommaListExpression, elements: readonly Expression[]): CommaListExpression;

  /* Expressions - Optional Chaining */
  createElementAccessChain(expression: Expression, questionDotToken: QuestionDotToken | undefined, index: number | Expression): ElementAccessChain;
  updateElementAccessChain(node: ElementAccessChain, expression: Expression, questionDotToken: QuestionDotToken | undefined, argumentExpression: Expression): ElementAccessChain;
  createCallChain(expression: Expression, questionDotToken: QuestionDotToken | undefined, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[] | undefined): CallChain;
  updateCallChain(node: CallChain, expression: Expression, questionDotToken: QuestionDotToken | undefined, typeArguments: readonly TypeNode[] | undefined, argumentsArray: readonly Expression[]): CallChain;
  createNonNullChain(expression: Expression): NonNullChain;
  updateNonNullChain(node: NonNullChain, expression: Expression): NonNullChain;

  /* Module & Import/Export */
  createModuleBlock(statements: readonly Statement[]): ModuleBlock;
  updateModuleBlock(node: ModuleBlock, statements: readonly Statement[]): ModuleBlock;
  createCaseBlock(clauses: readonly CaseOrDefaultClause[]): CaseBlock;
  updateCaseBlock(node: CaseBlock, clauses: readonly CaseOrDefaultClause[]): CaseBlock;
  createNamespaceExportDeclaration(name: string | Identifier): NamespaceExportDeclaration;
  updateNamespaceExportDeclaration(node: NamespaceExportDeclaration, name: Identifier): NamespaceExportDeclaration;
  createImportAttributes(elements: NodeArray<ImportAttribute>, multiLine?: boolean): ImportAttributes;
  updateImportAttributes(node: ImportAttributes, elements: NodeArray<ImportAttribute>, multiLine?: boolean): ImportAttributes;
  createImportAttribute(name: ImportAttributeName, value: Expression): ImportAttribute;
  updateImportAttribute(node: ImportAttribute, name: ImportAttributeName, value: Expression): ImportAttribute;
  createNamespaceImport(name: Identifier): NamespaceImport;
  updateNamespaceImport(node: NamespaceImport, name: Identifier): NamespaceImport;
  createNamespaceExport(name: ModuleExportName): NamespaceExport;
  updateNamespaceExport(node: NamespaceExport, name: ModuleExportName): NamespaceExport;
  createNamedImports(elements: readonly ImportSpecifier[]): NamedImports;
  updateNamedImports(node: NamedImports, elements: readonly ImportSpecifier[]): NamedImports;
  createImportSpecifier(isTypeOnly: boolean, propertyName: ModuleExportName | undefined, name: Identifier): ImportSpecifier;
  updateImportSpecifier(node: ImportSpecifier, isTypeOnly: boolean, propertyName: ModuleExportName | undefined, name: Identifier): ImportSpecifier;
  createNamedExports(elements: readonly ExportSpecifier[]): NamedExports;
  updateNamedExports(node: NamedExports, elements: readonly ExportSpecifier[]): NamedExports;
  createExportSpecifier(isTypeOnly: boolean, propertyName: string | ModuleExportName | undefined, name: string | ModuleExportName): ExportSpecifier;
  updateExportSpecifier(node: ExportSpecifier, isTypeOnly: boolean, propertyName: ModuleExportName | undefined, name: ModuleExportName): ExportSpecifier;
  createExternalModuleReference(expression: Expression): ExternalModuleReference;
  updateExternalModuleReference(node: ExternalModuleReference, expression: Expression): ExternalModuleReference;

  /* JSX */
  createJsxElement(openingElement: JsxOpeningElement, children: readonly JsxChild[], closingElement: JsxClosingElement): JsxElement;
  updateJsxElement(node: JsxElement, openingElement: JsxOpeningElement, children: readonly JsxChild[], closingElement: JsxClosingElement): JsxElement;
  createJsxSelfClosingElement(tagName: JsxTagNameExpression, typeArguments: readonly TypeNode[] | undefined, attributes: JsxAttributes): JsxSelfClosingElement;
  updateJsxSelfClosingElement(node: JsxSelfClosingElement, tagName: JsxTagNameExpression, typeArguments: readonly TypeNode[] | undefined, attributes: JsxAttributes): JsxSelfClosingElement;
  createJsxOpeningElement(tagName: JsxTagNameExpression, typeArguments: readonly TypeNode[] | undefined, attributes: JsxAttributes): JsxOpeningElement;
  updateJsxOpeningElement(node: JsxOpeningElement, tagName: JsxTagNameExpression, typeArguments: readonly TypeNode[] | undefined, attributes: JsxAttributes): JsxOpeningElement;
  createJsxClosingElement(tagName: JsxTagNameExpression): JsxClosingElement;
  updateJsxClosingElement(node: JsxClosingElement, tagName: JsxTagNameExpression): JsxClosingElement;
  createJsxFragment(openingFragment: JsxOpeningFragment, children: readonly JsxChild[], closingFragment: JsxClosingFragment): JsxFragment;
  updateJsxFragment(node: JsxFragment, openingFragment: JsxOpeningFragment, children: readonly JsxChild[], closingFragment: JsxClosingFragment): JsxFragment;
  createJsxText(text: string, containsOnlyTriviaWhiteSpaces?: boolean): JsxText;
  updateJsxText(node: JsxText, text: string, containsOnlyTriviaWhiteSpaces?: boolean): JsxText;
  createJsxOpeningFragment(): JsxOpeningFragment;
  createJsxJsxClosingFragment(): JsxClosingFragment;
  createJsxAttribute(name: JsxAttributeName, initializer: JsxAttributeValue | undefined): JsxAttribute;
  updateJsxAttribute(node: JsxAttribute, name: JsxAttributeName, initializer: JsxAttributeValue | undefined): JsxAttribute;
  createJsxAttributes(properties: readonly JsxAttributeLike[]): JsxAttributes;
  updateJsxAttributes(node: JsxAttributes, properties: readonly JsxAttributeLike[]): JsxAttributes;
  createJsxSpreadAttribute(expression: Expression): JsxSpreadAttribute;
  updateJsxSpreadAttribute(node: JsxSpreadAttribute, expression: Expression): JsxSpreadAttribute;
  createJsxExpression(dotDotDotToken: DotDotDotToken | undefined, expression: Expression | undefined): JsxExpression;
  updateJsxExpression(node: JsxExpression, expression: Expression | undefined): JsxExpression;
  createJsxNamespacedName(namespace: Identifier, name: Identifier): JsxNamespacedName;
  updateJsxNamespacedName(node: JsxNamespacedName, namespace: Identifier, name: Identifier): JsxNamespacedName;

  /* Clauses & Auxiliary Nodes */
  createCaseClause(expression: Expression, statements: readonly Statement[]): CaseClause;
  updateCaseClause(node: CaseClause, expression: Expression, statements: readonly Statement[]): CaseClause;
  createDefaultClause(statements: readonly Statement[]): DefaultClause;
  updateDefaultClause(node: DefaultClause, statements: readonly Statement[]): DefaultClause;
  createHeritageClause(token: HeritageClause["token"], types: readonly ExpressionWithTypeArguments[]): HeritageClause;
  updateHeritageClause(node: HeritageClause, types: readonly ExpressionWithTypeArguments[]): HeritageClause;
  createCatchClause(variableDeclaration: string | BindingName | VariableDeclaration | undefined, block: Block): CatchClause;
  updateCatchClause(node: CatchClause, variableDeclaration: VariableDeclaration | undefined, block: Block): CatchClause;
  createPropertyAssignment(name: string | PropertyName, initializer: Expression): PropertyAssignment;
  updatePropertyAssignment(node: PropertyAssignment, name: PropertyName, initializer: Expression): PropertyAssignment;
  createShorthandPropertyAssignment(name: string | Identifier, objectAssignmentInitializer?: Expression): ShorthandPropertyAssignment;
  updateShorthandPropertyAssignment(node: ShorthandPropertyAssignment, name: Identifier, objectAssignmentInitializer: Expression | undefined): ShorthandPropertyAssignment;
  createSpreadAssignment(expression: Expression): SpreadAssignment;
  updateSpreadAssignment(node: SpreadAssignment, expression: Expression): SpreadAssignment;
  createEnumMember(name: string | PropertyName, initializer?: Expression): EnumMember;
  updateEnumMember(node: EnumMember, name: PropertyName, initializer: Expression | undefined): EnumMember;

  /* JSDoc Nodes */
  createJSDocAllType(): JSDocAllType;
  createJSDocUnknownType(): JSDocUnknownType;
  createJSDocNonNullableType(type: TypeNode, postfix?: boolean): JSDocNonNullableType;
  updateJSDocNonNullableType(node: JSDocNonNullableType, type: TypeNode): JSDocNonNullableType;
  createJSDocNullableType(type: TypeNode, postfix?: boolean): JSDocNullableType;
  updateJSDocNullableType(node: JSDocNullableType, type: TypeNode): JSDocNullableType;
  createJSDocOptionalType(type: TypeNode): JSDocOptionalType;
  updateJSDocOptionalType(node: JSDocOptionalType, type: TypeNode): JSDocOptionalType;
  createJSDocFunctionType(parameters: readonly ParameterDeclaration[], type: TypeNode | undefined): JSDocFunctionType;
  updateJSDocFunctionType(node: JSDocFunctionType, parameters: readonly ParameterDeclaration[], type: TypeNode | undefined): JSDocFunctionType;
  createJSDocVariadicType(type: TypeNode): JSDocVariadicType;
  updateJSDocVariadicType(node: JSDocVariadicType, type: TypeNode): JSDocVariadicType;
  createJSDocNamepathType(type: TypeNode): JSDocNamepathType;
  updateJSDocNamepathType(node: JSDocNamepathType, type: TypeNode): JSDocNamepathType;
  createJSDocTypeExpression(type: TypeNode): JSDocTypeExpression;
  updateJSDocTypeExpression(node: JSDocTypeExpression, type: TypeNode): JSDocTypeExpression;
  createJSDocNameReference(name: EntityName | JSDocMemberName): JSDocNameReference;
  updateJSDocNameReference(node: JSDocNameReference, name: EntityName | JSDocMemberName): JSDocNameReference;
  createJSDocMemberName(left: EntityName | JSDocMemberName, right: Identifier): JSDocMemberName;
  updateJSDocMemberName(node: JSDocMemberName, left: EntityName | JSDocMemberName, right: Identifier): JSDocMemberName;
  createJSDocLink(name: EntityName | JSDocMemberName | undefined, text: string): JSDocLink;
  updateJSDocLink(node: JSDocLink, name: EntityName | JSDocMemberName | undefined, text: string): JSDocLink;
  createJSDocLinkCode(name: EntityName | JSDocMemberName | undefined, text: string): JSDocLinkCode;
  updateJSDocLinkCode(node: JSDocLinkCode, name: EntityName | JSDocMemberName | undefined, text: string): JSDocLinkCode;
  createJSDocLinkPlain(name: EntityName | JSDocMemberName | undefined, text: string): JSDocLinkPlain;
  updateJSDocLinkPlain(node: JSDocLinkPlain, name: EntityName | JSDocMemberName | undefined, text: string): JSDocLinkPlain;
  createJSDocTypeLiteral(jsDocPropertyTags?: readonly JSDocPropertyLikeTag[], isArrayType?: boolean): JSDocTypeLiteral;
  updateJSDocTypeLiteral(node: JSDocTypeLiteral, jsDocPropertyTags: readonly JSDocPropertyLikeTag[] | undefined, isArrayType: boolean | undefined): JSDocTypeLiteral;
  createJSDocSignature(typeParameters: readonly JSDocTemplateTag[] | undefined, parameters: readonly JSDocParameterTag[], type?: JSDocReturnTag): JSDocSignature;
  updateJSDocSignature(node: JSDocSignature, typeParameters: readonly JSDocTemplateTag[] | undefined, parameters: readonly JSDocParameterTag[], type: JSDocReturnTag | undefined): JSDocSignature;
  createJSDocTemplateTag(tagName: Identifier | undefined, constraint: JSDocTypeExpression | undefined, typeParameters: readonly TypeParameterDeclaration[], comment?: string | NodeArray<JSDocComment>): JSDocTemplateTag;
  updateJSDocTemplateTag(node: JSDocTemplateTag, tagName: Identifier | undefined, constraint: JSDocTypeExpression | undefined, typeParameters: readonly TypeParameterDeclaration[], comment: string | NodeArray<JSDocComment> | undefined): JSDocTemplateTag;
  createJSDocTypedefTag(tagName: Identifier | undefined, typeExpression?: JSDocTypeExpression | JSDocTypeLiteral, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string | NodeArray<JSDocComment>): JSDocTypedefTag;
  updateJSDocTypedefTag(node: JSDocTypedefTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression | JSDocTypeLiteral | undefined, fullName: Identifier | JSDocNamespaceDeclaration | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocTypedefTag;
  createJSDocParameterTag(tagName: Identifier | undefined, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, isNameFirst?: boolean, comment?: string | NodeArray<JSDocComment>): JSDocParameterTag;
  updateJSDocParameterTag(node: JSDocParameterTag, tagName: Identifier | undefined, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression | undefined, isNameFirst: boolean, comment: string | NodeArray<JSDocComment> | undefined): JSDocParameterTag;
  createJSDocPropertyTag(tagName: Identifier | undefined, name: EntityName, isBracketed: boolean, typeExpression?: JSDocTypeExpression, isNameFirst?: boolean, comment?: string | NodeArray<JSDocComment>): JSDocPropertyTag;
  updateJSDocPropertyTag(node: JSDocPropertyTag, tagName: Identifier | undefined, name: EntityName, isBracketed: boolean, typeExpression: JSDocTypeExpression | undefined, isNameFirst: boolean, comment: string | NodeArray<JSDocComment> | undefined): JSDocPropertyTag;
  createJSDocTypeTag(tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment?: string | NodeArray<JSDocComment>): JSDocTypeTag;
  updateJSDocTypeTag(node: JSDocTypeTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment: string | NodeArray<JSDocComment> | undefined): JSDocTypeTag;
  createJSDocSeeTag(tagName: Identifier | undefined, nameExpression: JSDocNameReference | undefined, comment?: string | NodeArray<JSDocComment>): JSDocSeeTag;
  updateJSDocSeeTag(node: JSDocSeeTag, tagName: Identifier | undefined, nameExpression: JSDocNameReference | undefined, comment?: string | NodeArray<JSDocComment>): JSDocSeeTag;
  createJSDocReturnTag(tagName: Identifier | undefined, typeExpression?: JSDocTypeExpression, comment?: string | NodeArray<JSDocComment>): JSDocReturnTag;
  updateJSDocReturnTag(node: JSDocReturnTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocReturnTag;
  createJSDocThisTag(tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment?: string | NodeArray<JSDocComment>): JSDocThisTag;
  updateJSDocThisTag(node: JSDocThisTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocThisTag;
  createJSDocEnumTag(tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment?: string | NodeArray<JSDocComment>): JSDocEnumTag;
  updateJSDocEnumTag(node: JSDocEnumTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment: string | NodeArray<JSDocComment> | undefined): JSDocEnumTag;
  createJSDocCallbackTag(tagName: Identifier | undefined, typeExpression: JSDocSignature, fullName?: Identifier | JSDocNamespaceDeclaration, comment?: string | NodeArray<JSDocComment>): JSDocCallbackTag;
  updateJSDocCallbackTag(node: JSDocCallbackTag, tagName: Identifier | undefined, typeExpression: JSDocSignature, fullName: Identifier | JSDocNamespaceDeclaration | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocCallbackTag;
  createJSDocOverloadTag(tagName: Identifier | undefined, typeExpression: JSDocSignature, comment?: string | NodeArray<JSDocComment>): JSDocOverloadTag;
  updateJSDocOverloadTag(node: JSDocOverloadTag, tagName: Identifier | undefined, typeExpression: JSDocSignature, comment: string | NodeArray<JSDocComment> | undefined): JSDocOverloadTag;
  createJSDocAugmentsTag(tagName: Identifier | undefined, className: JSDocAugmentsTag["class"], comment?: string | NodeArray<JSDocComment>): JSDocAugmentsTag;
  updateJSDocAugmentsTag(node: JSDocAugmentsTag, tagName: Identifier | undefined, className: JSDocAugmentsTag["class"], comment: string | NodeArray<JSDocComment> | undefined): JSDocAugmentsTag;
  createJSDocImplementsTag(tagName: Identifier | undefined, className: JSDocImplementsTag["class"], comment?: string | NodeArray<JSDocComment>): JSDocImplementsTag;
  updateJSDocImplementsTag(node: JSDocImplementsTag, tagName: Identifier | undefined, className: JSDocImplementsTag["class"], comment: string | NodeArray<JSDocComment> | undefined): JSDocImplementsTag;
  createJSDocAuthorTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocAuthorTag;
  updateJSDocAuthorTag(node: JSDocAuthorTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocAuthorTag;
  createJSDocClassTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocClassTag;
  updateJSDocClassTag(node: JSDocClassTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocClassTag;
  createJSDocPublicTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocPublicTag;
  updateJSDocPublicTag(node: JSDocPublicTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocPublicTag;
  createJSDocPrivateTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocPrivateTag;
  updateJSDocPrivateTag(node: JSDocPrivateTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocPrivateTag;
  createJSDocProtectedTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocProtectedTag;
  updateJSDocProtectedTag(node: JSDocProtectedTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocProtectedTag;
  createJSDocReadonlyTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocReadonlyTag;
  updateJSDocReadonlyTag(node: JSDocReadonlyTag, tagName: Identifier | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocReadonlyTag;
  createJSDocUnknownTag(tagName: Identifier, comment?: string | NodeArray<JSDocComment>): JSDocUnknownTag;
  updateJSDocUnknownTag(node: JSDocUnknownTag, tagName: Identifier, comment: string | NodeArray<JSDocComment> | undefined): JSDocUnknownTag;
  createJSDocDeprecatedTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocDeprecatedTag;
  updateJSDocDeprecatedTag(node: JSDocDeprecatedTag, tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocDeprecatedTag;
  createJSDocOverrideTag(tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocOverrideTag;
  updateJSDocOverrideTag(node: JSDocOverrideTag, tagName: Identifier | undefined, comment?: string | NodeArray<JSDocComment>): JSDocOverrideTag;
  createJSDocThrowsTag(tagName: Identifier, typeExpression: JSDocTypeExpression | undefined, comment?: string | NodeArray<JSDocComment>): JSDocThrowsTag;
  updateJSDocThrowsTag(node: JSDocThrowsTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression | undefined, comment?: string | NodeArray<JSDocComment> | undefined): JSDocThrowsTag;
  createJSDocSatisfiesTag(tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment?: string | NodeArray<JSDocComment>): JSDocSatisfiesTag;
  updateJSDocSatisfiesTag(node: JSDocSatisfiesTag, tagName: Identifier | undefined, typeExpression: JSDocTypeExpression, comment: string | NodeArray<JSDocComment> | undefined): JSDocSatisfiesTag;
  createJSDocImportTag(tagName: Identifier | undefined, importClause: ImportClause | undefined, moduleSpecifier: Expression, attributes?: ImportAttributes, comment?: string | NodeArray<JSDocComment>): JSDocImportTag;
  updateJSDocImportTag(node: JSDocImportTag, tagName: Identifier | undefined, importClause: ImportClause | undefined, moduleSpecifier: Expression, attributes: ImportAttributes | undefined, comment: string | NodeArray<JSDocComment> | undefined): JSDocImportTag;
  createJSDocText(text: string): JSDocText;
  updateJSDocText(node: JSDocText, text: string): JSDocText;
  createJSDocComment(comment?: string | NodeArray<JSDocComment> | undefined, tags?: readonly JSDocTag[] | undefined): JSDoc;
  updateJSDocComment(node: JSDoc, comment: string | NodeArray<JSDocComment> | undefined, tags: readonly JSDocTag[] | undefined): JSDoc;

  /* Helper & Convenience Methods */
  createComma(left: Expression, right: Expression): BinaryExpression;
  createAssignment(left: Expression, right: Expression): AssignmentExpression<EqualsToken>;
  createLogicalOr(left: Expression, right: Expression): BinaryExpression;
  createLogicalAnd(left: Expression, right: Expression): BinaryExpression;
  createBitwiseOr(left: Expression, right: Expression): BinaryExpression;
  createBitwiseXor(left: Expression, right: Expression): BinaryExpression;
  createBitwiseAnd(left: Expression, right: Expression): BinaryExpression;
  createStrictEquality(left: Expression, right: Expression): BinaryExpression;
  createStrictInequality(left: Expression, right: Expression): BinaryExpression;
  createEquality(left: Expression, right: Expression): BinaryExpression;
  createInequality(left: Expression, right: Expression): BinaryExpression;
  createLessThan(left: Expression, right: Expression): BinaryExpression;
  createLessThanEquals(left: Expression, right: Expression): BinaryExpression;
  createGreaterThan(left: Expression, right: Expression): BinaryExpression;
  createGreaterThanEquals(left: Expression, right: Expression): BinaryExpression;
  createLeftShift(left: Expression, right: Expression): BinaryExpression;
  createRightShift(left: Expression, right: Expression): BinaryExpression;
  createUnsignedRightShift(left: Expression, right: Expression): BinaryExpression;
  createAdd(left: Expression, right: Expression): BinaryExpression;
  createSubtract(left: Expression, right: Expression): BinaryExpression;
  createMultiply(left: Expression, right: Expression): BinaryExpression;
  createDivide(left: Expression, right: Expression): BinaryExpression;
  createModulo(left: Expression, right: Expression): BinaryExpression;
  createExponent(left: Expression, right: Expression): BinaryExpression;
  createPrefixPlus(operand: Expression): PrefixUnaryExpression;
  createPrefixMinus(operand: Expression): PrefixUnaryExpression;
  createPrefixIncrement(operand: Expression): PrefixUnaryExpression;
  createPrefixDecrement(operand: Expression): PrefixUnaryExpression;
  createBitwiseNot(operand: Expression): PrefixUnaryExpression;
  createLogicalNot(operand: Expression): PrefixUnaryExpression;
  createPostfixIncrement(operand: Expression): PostfixUnaryExpression;
  createPostfixDecrement(operand: Expression): PostfixUnaryExpression;
  createImmediatelyInvokedFunctionExpression(statements: readonly Statement[]): CallExpression;
  createImmediatelyInvokedFunctionExpression(statements: readonly Statement[], param: ParameterDeclaration, paramValue: Expression): CallExpression;
  createImmediatelyInvokedArrowFunction(statements: readonly Statement[]): ImmediatelyInvokedArrowFunction;
  createImmediatelyInvokedArrowFunction(statements: readonly Statement[], param: ParameterDeclaration, paramValue: Expression): ImmediatelyInvokedArrowFunction;
  createVoidZero(): VoidExpression;
  createExportDefault(expression: Expression): ExportAssignment;
  createExternalModuleExport(exportName: Identifier): ExportDeclaration;
  restoreOuterExpressions(outerExpression: Expression | undefined, innerExpression: Expression, kinds?: OuterExpressionKinds): Expression;

  /* Node Replacement Methods */
  replaceModifiers<T extends HasModifiers>(node: T, modifiers: readonly Modifier[] | ModifierFlags | undefined): T;
  replaceDecoratorsAndModifiers<T extends HasModifiers & HasDecorators>(node: T, modifiers: readonly ModifierLike[] | undefined): T;
  replacePropertyName<T extends AccessorDeclaration | MethodDeclaration | MethodSignature | PropertyDeclaration | PropertySignature | PropertyAssignment>(node: T, name: T["name"]): T;

  /* Bundles */
  createBundle(sourceFiles: readonly SourceFile[]): Bundle;
  updateBundle(node: Bundle, sourceFiles: readonly SourceFile[]): Bundle;

  /* Node Arrays */
  createNodeArray<T extends Node>(elements?: readonly T[], hasTrailingComma?: boolean): NodeArray<T>;
}
```

### Usage Example

Creating a simple function declaration:

```typescript
import * as ts from 'typescript';

const { factory } = ts;

// Create: function add(a: number, b: number): number { return a + b; }
const functionDecl = factory.createFunctionDeclaration(
  undefined, // modifiers
  undefined, // asteriskToken
  'add', // name
  undefined, // typeParameters
  [
    factory.createParameterDeclaration(
      undefined, // modifiers
      undefined, // dotDotDotToken
      'a', // name
      undefined, // questionToken
      factory.createKeywordTypeNode(ts.SyntaxKind.NumberKeyword) // type
    ),
    factory.createParameterDeclaration(
      undefined,
      undefined,
      'b',
      undefined,
      factory.createKeywordTypeNode(ts.SyntaxKind.NumberKeyword)
    )
  ],
  factory.createKeywordTypeNode(ts.SyntaxKind.NumberKeyword), // return type
  factory.createBlock([
    factory.createReturnStatement(
      factory.createBinaryExpression(
        factory.createIdentifier('a'),
        ts.SyntaxKind.PlusToken,
        factory.createIdentifier('b')
      )
    )
  ])
);

// Print the created node
const printer = ts.createPrinter();
const sourceFile = ts.createSourceFile('temp.ts', '', ts.ScriptTarget.Latest);
const output = printer.printNode(ts.EmitHint.Unspecified, functionDecl, sourceFile);
console.log(output);
```
