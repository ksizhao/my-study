package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-05-29 20:36
 */
public class Test {

    public static void main(String[] args) {
        CarStore factory=new CarStore();
        Car car=factory.manufactureCar("benz");
        String des=car.getDescription();
        System.out.println(des);
        car.getBrand();
        Car car1=factory.manufactureCar("bmw");
        String des1=car1.getDescription();
        System.out.println(des1);
        car1.getBrand();

        Car car2=factory.manufactureCar("audi");
        car2.getBrand();
        System.out.println(car2.cycle.getDescription());
        System.out.println(car2.glass.getDescription());
    }
}
