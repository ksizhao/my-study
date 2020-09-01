package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 奔驰工厂
 * @create 2019-05-29 20:28
 */
public class CarStore extends  CarFactory{

    Car car;
    MaterialFactory chineseFactory=new ChineseFactory();

    @Override
    Car manufactureCar(String carType) {

        if ("benz".equals(carType)) {
            car=new BenzCar();
        }
        if ("bmw".equals(carType)) {
            car=new BmwCar();
        }
        if ("audi".equals(carType)) {
            car=new AudiCar(chineseFactory);
        }
        return car;
    }
}
