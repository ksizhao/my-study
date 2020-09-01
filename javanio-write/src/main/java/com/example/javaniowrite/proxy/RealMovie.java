package com.example.javaniowrite.proxy;

/**
 * @author zhaoliancan
 * @description 电影真实类
 * @create 2019-01-03 19:09
 */
public class RealMovie implements Movie {

    @Override
    public void play() {
        System.out.println("我正在播放《钢铁侠》");
    }
}
