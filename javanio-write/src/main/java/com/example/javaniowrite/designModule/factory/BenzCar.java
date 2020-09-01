package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 奔驰汽车
 * @create 2019-05-29 20:31
 */
public class BenzCar extends Car {


    public BenzCar () {
        description="我是奔驰汽车";
    }
    @Override
    void getBrand() {
        System.out.println("奔驰汽车由奔驰工厂制造");
    }
}
