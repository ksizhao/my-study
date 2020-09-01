package com.example.javaniowrite.designModule.adapter;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-06-10 16:18
 */
public class Test {

    public static void main(String[] args) {
        Ac ac=new Ac();
        ac.getElectricityType();
        Dc dc=new Dc();
        dc.description();

        DcElectricity dcAdapter=new DcAdapter(ac);
        dcAdapter.description();
    }
}
