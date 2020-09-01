package com.example.javaniowrite.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

/**
 * @author zhaoliancan
 * @description 巨幕
 * @create 2019-01-03 19:53
 */
public class Imax  implements InvocationHandler {

    private Object movie;


    public  Imax (Object movie) {

        this.movie=movie;

    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {

        System.out.println("电影开始，电影屏幕是："+this.getClass().getSimpleName());
        method.invoke(movie,args);
        System.out.println("电影结束");
        return null;
    }
}
