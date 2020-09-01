package com.example.javaniowrite.designModule.beverage;

/**
 * @author zhaoliancan
 * @description 装饰者模式
 * @create 2019-05-28 13:41
 */
public abstract class Beverage {

    String description="Unknown Beverage";

    public String getDescription() {

        return description;
    }

    public abstract  double cost();
}
