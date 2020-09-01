package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 狗
 * @create 2019-02-12 16:57
 */
public class Dog implements Animal {

    @Override
    public void getAnimalType() {
        System.out.println("我是一只哈巴狗");
    }
}
