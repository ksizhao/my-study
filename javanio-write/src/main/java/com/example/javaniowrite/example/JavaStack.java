package com.example.javaniowrite.example;

import java.rmi.server.UID;
import java.util.UUID;

/**
 * @author zhaoliancan
 * @description 栈溢出
 * @create 2019-07-03 14:49
 */
public class JavaStack {

    private  int stackLength = 1;

    private void stackCount() {
        stackLength++;
        stackCount();
        System.out.println(stackLength);
        UUID rek=UUID.randomUUID();
    }


    public static void main(String[] args) {

//        JavaStack javaStack=new JavaStack();
//        try {
//
//            javaStack.stackCount();
//        } catch (Exception e) {
//            e.printStackTrace();
//            System.out.println("栈深度"+javaStack.stackLength);
//            throw  e;
//
//        }
        UUID rek=UUID.randomUUID();
        System.out.println(rek);
    }
}
