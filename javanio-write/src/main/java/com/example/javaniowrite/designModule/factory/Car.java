package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 汽车
 * @create 2019-05-29 20:24
 */
public abstract class Car {

    String description;
    // 玻璃
    Glass glass;
    // 轮胎
    Cycle cycle;


    public String getDescription() {
        return description;
    }

    abstract void getBrand();
}
