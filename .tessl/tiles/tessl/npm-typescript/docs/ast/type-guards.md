# Type Guards

Comprehensive type guard functions for all AST node types.

## Type Guard Functions

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

## Usage Example

```typescript
import * as ts from 'typescript';

function visit(node: ts.Node) {
  if (ts.isFunctionDeclaration(node)) {
    console.log('Found function:', node.name?.getText());
  } else if (ts.isVariableDeclaration(node)) {
    console.log('Found variable:', node.name.getText());
  } else if (ts.isCallExpression(node)) {
    console.log('Found call expression');
  }
  
  ts.forEachChild(node, visit);
}

const sourceFile = ts.createSourceFile(
  'example.ts',
  'function foo() { const x = 1; }',
  ts.ScriptTarget.Latest,
  true
);

visit(sourceFile);
```

