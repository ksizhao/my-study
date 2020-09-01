package com.example.javaniowrite.designModule.statrage;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-05-22 19:09
 */
public class Test {

    public static void main(String[] args) {


        Character character=new Adc(new BowAndArrow());
        Character character1=new Ap(new Sword());
        Character character2=new Troll();


        character2.fight();
        character2.useWeapon();

        character.description();
        character.fight();
        character.useWeapon();
        System.out.println("更换武器");
        character.setWeapon(new Sword());
        character.useWeapon();
        character1.description();
        character1.fight();
        character1.useWeapon();
    }
}
