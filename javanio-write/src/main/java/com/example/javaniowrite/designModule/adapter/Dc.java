package com.example.javaniowrite.designModule.adapter;

/**
 * @author zhaoliancan
 * @description 直流电
 * @create 2019-06-10 16:14
 */
public class Dc implements DcElectricity {

    @Override
    public void description() {
        System.out.println("我是直流电");
    }
}
