package com.example.javaniowrite.simpleton;

/**
 * @author zhaoliancan
 * @description 饿汉模式
 * @create 2019-01-03 16:33
 */
public class SimpleLon {

    private static SimpleLon simpleLon=new SimpleLon();
    private SimpleLon() {}

    public static SimpleLon getInstance(){

        return simpleLon;
    }
}
