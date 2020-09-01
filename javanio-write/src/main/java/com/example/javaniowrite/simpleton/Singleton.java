package com.example.javaniowrite.simpleton;

import java.util.HashMap;

/**
 * @author zhaoliancan
 * @description 静态内部类单例
 * @create 2019-01-03 16:42
 */
public class Singleton {

    private Singleton () { }



    public static  final  Singleton getInstance () {

        return   SingletonSholder.INSTANCE;
    }

    /**
     * 静态内部类
     */
    private static class SingletonSholder {

        private  static  final Singleton INSTANCE=new Singleton();


    }


}
