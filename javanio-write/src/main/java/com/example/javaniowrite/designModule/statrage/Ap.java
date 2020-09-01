package com.example.javaniowrite.designModule.statrage;

/**
 * @author zhaoliancan
 * @description 法师
 * @create 2019-05-22 19:05
 */
public class Ap extends Character {

    @Override
    void fight() {
        System.out.println("我是法师");
    }

    public Ap(Weapon weapon) {
        this.weapon=weapon;
    }
}
