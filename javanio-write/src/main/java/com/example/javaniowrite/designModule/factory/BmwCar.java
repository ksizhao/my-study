package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 宝马汽车
 * @create 2019-05-29 20:33
 */
public class BmwCar extends Car {
    @Override
    void getBrand() {
        System.out.println("宝马汽车由宝马汽车工厂制造");
    }

    public BmwCar() {
        description="我是宝马汽车";
    }
}
