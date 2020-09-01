package com.example.javaniowrite.designModule.beverage;

/**
 * @author zhaoliancan
 * @description 混合模式
 * @create 2019-05-28 13:47
 */
public class HouseBlend extends Beverage{


    public HouseBlend () {
        description="HouseBlend";
    }
    @Override
    public double cost() {
        return 0.50;
    }
}
