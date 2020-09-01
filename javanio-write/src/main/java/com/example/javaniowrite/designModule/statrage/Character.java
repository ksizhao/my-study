package com.example.javaniowrite.designModule.statrage;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-05-22 18:58
 */
public abstract class Character {


    Weapon weapon;


    abstract void fight();

    public Weapon getWeapon() {
        return weapon;
    }

    public void setWeapon(Weapon weapon) {
        this.weapon = weapon;
    }

    public  void description() {
        System.out.println("游戏角色:");
    }

    /**
     * 委托给行为类
     */
    public  void  useWeapon() {
        weapon.weaponType();
    }
}
