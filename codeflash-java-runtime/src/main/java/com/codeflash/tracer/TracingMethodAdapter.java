package com.codeflash.tracer;

import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.AdviceAdapter;

/**
 * ASM AdviceAdapter that captures method arguments on entry.
 *
 * <p>On method entry, boxes all parameters into an Object[] array and calls
 * {@link TraceRecorder#onEntry} to record the invocation. For instance methods,
 * {@code this} is included as the first element.
 */
public class TracingMethodAdapter extends AdviceAdapter {

    private static final String TRACE_RECORDER = "com/codeflash/tracer/TraceRecorder";

    private final String className;
    private final String methodName;
    private final String descriptor;
    private final int lineNumber;
    private final String sourceFile;
    private final boolean isStatic;

    protected TracingMethodAdapter(MethodVisitor mv, int access, String name, String descriptor,
                                   String className, int lineNumber, String sourceFile) {
        super(Opcodes.ASM9, mv, access, name, descriptor);
        this.className = className;
        this.methodName = name;
        this.descriptor = descriptor;
        this.lineNumber = lineNumber;
        this.sourceFile = sourceFile;
        this.isStatic = (access & Opcodes.ACC_STATIC) != 0;
    }

    @Override
    protected void onMethodEnter() {
        // Build Object[] containing explicit parameters only (skip 'this' to avoid
        // expensive serialization of the receiver's full object graph)
        Type[] argTypes = Type.getArgumentTypes(descriptor);

        // Push array size and create Object[]
        pushInt(argTypes.length);
        mv.visitTypeInsn(ANEWARRAY, "java/lang/Object");

        int arrayIndex = 0;
        int localIndex = isStatic ? 0 : 1; // skip 'this' slot for instance methods

        // Box and store each parameter
        for (Type argType : argTypes) {
            mv.visitInsn(DUP);
            pushInt(arrayIndex);
            loadAndBox(argType, localIndex);
            mv.visitInsn(AASTORE);
            arrayIndex++;
            localIndex += argType.getSize();
        }

        // Stack now has: Object[] args on top
        // Store in a local variable
        int argsLocal = newLocal(Type.getType("[Ljava/lang/Object;"));
        mv.visitVarInsn(ASTORE, argsLocal);

        // Call TraceRecorder.getInstance().onEntry(className, methodName, descriptor, lineNumber, sourceFile, args)
        mv.visitMethodInsn(INVOKESTATIC, TRACE_RECORDER, "getInstance",
                "()L" + TRACE_RECORDER + ";", false);
        mv.visitLdcInsn(className.replace('/', '.'));
        mv.visitLdcInsn(methodName);
        mv.visitLdcInsn(descriptor);
        pushInt(lineNumber);
        mv.visitLdcInsn(sourceFile != null ? sourceFile : "");
        mv.visitVarInsn(ALOAD, argsLocal);
        mv.visitMethodInsn(INVOKEVIRTUAL, TRACE_RECORDER, "onEntry",
                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;[Ljava/lang/Object;)V",
                false);
    }

    private void loadAndBox(Type type, int localIndex) {
        switch (type.getSort()) {
            case Type.BOOLEAN:
                mv.visitVarInsn(ILOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Boolean", "valueOf", "(Z)Ljava/lang/Boolean;", false);
                break;
            case Type.BYTE:
                mv.visitVarInsn(ILOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Byte", "valueOf", "(B)Ljava/lang/Byte;", false);
                break;
            case Type.CHAR:
                mv.visitVarInsn(ILOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Character", "valueOf", "(C)Ljava/lang/Character;", false);
                break;
            case Type.SHORT:
                mv.visitVarInsn(ILOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Short", "valueOf", "(S)Ljava/lang/Short;", false);
                break;
            case Type.INT:
                mv.visitVarInsn(ILOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", false);
                break;
            case Type.LONG:
                mv.visitVarInsn(LLOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Long", "valueOf", "(J)Ljava/lang/Long;", false);
                break;
            case Type.FLOAT:
                mv.visitVarInsn(FLOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Float", "valueOf", "(F)Ljava/lang/Float;", false);
                break;
            case Type.DOUBLE:
                mv.visitVarInsn(DLOAD, localIndex);
                mv.visitMethodInsn(INVOKESTATIC, "java/lang/Double", "valueOf", "(D)Ljava/lang/Double;", false);
                break;
            default:
                // Object or array — just load reference
                mv.visitVarInsn(ALOAD, localIndex);
                break;
        }
    }

    private void pushInt(int value) {
        if (value >= -1 && value <= 5) {
            mv.visitInsn(ICONST_0 + value);
        } else if (value >= Byte.MIN_VALUE && value <= Byte.MAX_VALUE) {
            mv.visitIntInsn(BIPUSH, value);
        } else if (value >= Short.MIN_VALUE && value <= Short.MAX_VALUE) {
            mv.visitIntInsn(SIPUSH, value);
        } else {
            mv.visitLdcInsn(value);
        }
    }
}
