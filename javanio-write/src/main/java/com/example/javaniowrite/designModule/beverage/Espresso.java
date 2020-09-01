package com.example.javaniowrite.designModule.beverage;

/**
 * @author zhaoliancan
 * @description 咖啡种类
 * @create 2019-05-28 13:46
 */
public class Espresso extends Beverage {


    public Espresso () {
        description="Espresso";
    }

    @Override
    public double cost() {
        return 0.86;
    }
}
