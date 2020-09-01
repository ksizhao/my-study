package com.example.javaniowrite.designModule.adapter;

/**
 * @author zhaoliancan
 * @description 交流电
 * @create 2019-06-10 16:13
 */
public class Ac implements  AcElectricity{

    @Override
    public void getElectricityType() {
        System.out.println("我是交流电");
    }
}
