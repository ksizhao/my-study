package com.example.javaniowrite.designModule.factory;

/**
 * @author zhaoliancan
 * @description 中国工厂
 * @create 2019-05-30 19:43
 */
public class ChineseFactory implements MaterialFactory {


    @Override
    public Glass createGlass() {
        return new Glass();
    }

    @Override
    public Cycle createCycle() {
        return new Cycle();
    }
}
