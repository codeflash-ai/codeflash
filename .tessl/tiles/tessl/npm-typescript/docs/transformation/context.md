# Transformation Context

Context object provided to transformers with factory access and lifecycle management.

## Transformation Context Interface

```typescript { .api }
interface CoreTransformationContext {
  readonly factory: NodeFactory;
  getCompilerOptions(): CompilerOptions;
  startLexicalEnvironment(): void;
  suspendLexicalEnvironment(): void;
  resumeLexicalEnvironment(): void;
  endLexicalEnvironment(): Statement[] | undefined;
  hoistFunctionDeclaration(node: FunctionDeclaration): void;
  hoistVariableDeclaration(node: Identifier): void;
}

interface TransformationContext extends CoreTransformationContext {
  requestEmitHelper(helper: EmitHelper): void;
  readEmitHelpers(): EmitHelper[] | undefined;
  enableSubstitution(kind: SyntaxKind): void;
  isSubstitutionEnabled(node: Node): boolean;
  onSubstituteNode: (hint: EmitHint, node: Node) => Node;
  enableEmitNotification(kind: SyntaxKind): void;
  isEmitNotificationEnabled(node: Node): boolean;
  onEmitNode: (
    hint: EmitHint,
    node: Node,
    emitCallback: (hint: EmitHint, node: Node) => void
  ) => void;
}
```

## Usage Example

```typescript
import * as ts from 'typescript';

function transformer(): ts.TransformerFactory<ts.SourceFile> {
  return (context: ts.TransformationContext) => {
    return (sourceFile: ts.SourceFile) => {
      const visitor = (node: ts.Node): ts.Node => {
        if (ts.isFunctionDeclaration(node)) {
          // Use context.factory to create new nodes
          const newName = context.factory.createIdentifier(
            node.name?.text.toUpperCase() || 'ANONYMOUS'
          );
          
          // Use context methods for lexical environment management
          context.startLexicalEnvironment();
          
          // Transform function body
          const newBody = ts.visitEachChild(node.body!, visitor, context);
          
          // Get hoisted declarations
          const hoisted = context.endLexicalEnvironment();
          
          return context.factory.updateFunctionDeclaration(
            node,
            node.modifiers,
            node.asteriskToken,
            newName,
            node.typeParameters,
            node.parameters,
            node.type,
            newBody
          );
        }
        
        return ts.visitEachChild(node, visitor, context);
      };
      
      return ts.visitNode(sourceFile, visitor) as ts.SourceFile;
    };
  };
}
```

