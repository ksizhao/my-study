package com.example.javaniowrite.designModule.statrage;

import com.example.javaniowrite.designModule.statrage.Character;
import com.example.javaniowrite.designModule.statrage.Sword;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-05-23 10:35
 */
public class Troll extends Character {

    @Override
    void fight() {

        System.out.println("我是妖怪，会使用法术");

    }

    public Troll(){
        weapon=new Sword();
    }
}
