package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 奥迪汽车
 * @create 2019-05-30 19:49
 */
public class AudiCar extends Car{
    MaterialFactory materialFactory;

    public AudiCar(MaterialFactory materialFactory) {

        this.materialFactory=materialFactory;

    }
    @Override
    void getBrand() {
        System.out.println("我是灯厂，奥迪汽车");
        glass=materialFactory.createGlass();
        cycle=materialFactory.createCycle();
    }


}
