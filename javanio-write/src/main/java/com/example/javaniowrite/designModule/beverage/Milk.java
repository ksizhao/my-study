package com.example.javaniowrite.designModule.beverage;

/**
 * @author zhaoliancan
 * @description 牛奶装饰类
 * @create 2019-05-28 13:48
 */
public class Milk  extends Conditment{

    Beverage beverage;


    public Milk(Beverage beverage) {
        this.beverage=beverage;
    }


    @Override
    public String getDescription() {
        return beverage.getDescription()+"Milk";
    }



    @Override
    public double cost() {
        return beverage.cost()+0.40;
    }
}
