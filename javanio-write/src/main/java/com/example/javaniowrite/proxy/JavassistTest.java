package com.example.javaniowrite.proxy;

import javassist.*;

import java.io.IOException;

/**
 * @author zhaoliancan
 * @version 1.0.0
 * @ClassName JavassistTest.java
 * @Description TODO
 * @createTime 2019年09月09日 17:53:00
 */
public class JavassistTest {

    /**
     * 字节码增强
     * @param args
     */
    public static void main(String[] args) {

        ClassPool cp=ClassPool.getDefault();
        try {
            CtClass ctClass=cp.get("com.example.javaniowrite.proxy.Base");
            CtMethod ctMethod=ctClass.getDeclaredMethod("process");
            ctMethod.insertBefore("{ System.out.println(\"start\"); }");
            ctMethod.insertAfter("{ System.out.println(\"end\"); }");
            Class c=ctClass.toClass();
            ctClass.writeFile("D:\\projects\\msp");
            Base newBase= (Base) c.newInstance();
            newBase.process();
        } catch (NotFoundException | CannotCompileException | InstantiationException | IllegalAccessException | IOException e) {
            e.printStackTrace();
        }
    }
}
