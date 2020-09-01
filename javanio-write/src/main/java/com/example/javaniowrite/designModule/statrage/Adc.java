package com.example.javaniowrite.designModule.statrage;

/**
 * @author zhaoliancan
 * @description 射手
 * @create 2019-05-22 19:03
 */
public class Adc extends Character {

    @Override
    void fight() {
        System.out.println("我是ADC远程输出");
    }

    public Adc(Weapon weapon) {
        this.weapon=weapon;
    }
}
