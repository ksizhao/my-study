package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 猫工厂
 * @create 2019-02-12 17:03
 */
public class CutFactory  implements Factory{

    @Override
    public Animal getAnimal() {
        return new Cut();
    }


    @Override
    public Action getAction() {
        return new CutAction();
    }
}
