package com.example.javaniowrite.simpleton;

/**
 * @author zhaoliancan
 * @description 单例模式,懒汉模式
 * @create 2019-01-03 15:59
 */
public class SimpleExample {



    private static SimpleExample simpleExample=null;

    private SimpleExample() {}

    /**
     * 懒汉模式，线程安全
     * @return
     */
    public static synchronized SimpleExample getExample(){

        if(simpleExample==null){

            simpleExample=new SimpleExample();
        }

        return simpleExample;
    }

    public static SimpleExample getsingle() {
        return simpleExample=new SimpleExample();
    }


    /**
     * DCL单例（双重检查锁定）
     * @return
     */
    public static  SimpleExample getExample1(){

        if (simpleExample==null) {

            synchronized (SimpleExample.class) {
                if (simpleExample==null) {
                    simpleExample=new SimpleExample();
                }
            }
        }

        return simpleExample;
    }


}



