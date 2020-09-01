package com.example.javaniowrite.designModule.beverage;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-05-28 13:52
 */
public class Test {

    public static void main(String[] args) {

        Beverage beverage1=new HouseBlend();
        System.out.println("咖啡种类"+beverage1.getDescription()+"价格"+beverage1.cost());
        Beverage beverage=new Espresso();
        // 牛奶咖啡
        beverage=new Milk(beverage);
        //beverage=new Milk(beverage);
        System.out.println("咖啡种类"+beverage.getDescription()+"价格"+beverage.cost());
    }
}
