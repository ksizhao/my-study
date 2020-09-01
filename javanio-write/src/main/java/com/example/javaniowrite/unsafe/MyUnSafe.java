package com.example.javaniowrite.unsafe;

import sun.misc.Unsafe;

import java.lang.reflect.Field;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年06月03日 15:26:00
 */
public class MyUnSafe {

    public static void main(String[] args) {

        Unsafe unsafe = getUnsafe();
        unsafe.allocateMemory(10000L);
    }

    private static Unsafe getUnsafe() {

        try {
            Field filed = Field.class.getDeclaredField("theUnsafe");
            filed.setAccessible(true);
            try {
                return (Unsafe) filed.get(null);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        }
        return null;

    }
}
