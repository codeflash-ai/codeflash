package com.codeflash.tracer;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.util.Collections;
import java.util.Map;

public class TracingClassVisitor extends ClassVisitor {

    private final String internalClassName;
    private final Map<String, Integer> methodLineNumbers;
    private String sourceFile;

    public TracingClassVisitor(ClassVisitor classVisitor, String internalClassName,
                               Map<String, Integer> methodLineNumbers) {
        super(Opcodes.ASM9, classVisitor);
        this.internalClassName = internalClassName;
        this.methodLineNumbers = methodLineNumbers != null ? methodLineNumbers : Collections.emptyMap();
    }

    @Override
    public void visitSource(String source, String debug) {
        super.visitSource(source, debug);
        this.sourceFile = source;
    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String descriptor,
                                     String signature, String[] exceptions) {
        MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);

        // Skip static initializers, synthetic, and bridge methods
        if (name.equals("<clinit>")
                || (access & Opcodes.ACC_SYNTHETIC) != 0
                || (access & Opcodes.ACC_BRIDGE) != 0) {
            return mv;
        }

        // Skip constructors for now (they have complex init semantics)
        if (name.equals("<init>")) {
            return mv;
        }

        int lineNumber = methodLineNumbers.getOrDefault(name + descriptor, 0);
        return new TracingMethodAdapter(mv, access, name, descriptor,
                internalClassName, lineNumber, sourceFile != null ? sourceFile : "");
    }
}
