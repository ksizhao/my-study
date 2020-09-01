package com.example.javaniowrite.proxy;

/**
 * @author zhaoliancan
 * @description 巨幕
 * @create 2019-01-03 20:42
 */
public class ImaxMovie implements Movie {

    @Override
    public void play() {
        System.out.println("我现在放的是《蜘蛛侠》");
    }
}
