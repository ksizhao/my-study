package com.example.javaniowrite.designModule.adapter;

/**
 * @author zhaoliancan
 * @description 直流电适配器
 * @create 2019-06-10 16:15
 */
public class DcAdapter implements DcElectricity {

    AcElectricity acElectricity;
    @Override
    public void description() {
        acElectricity.getElectricityType();
    }

    public DcAdapter(AcElectricity acElectricity) {
        this.acElectricity=acElectricity;

    }


}
