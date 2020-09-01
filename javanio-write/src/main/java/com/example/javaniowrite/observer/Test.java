package com.example.javaniowrite.observer;

import javax.swing.*;
import java.util.Observer;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-05-23 13:43
 */
public class Test {

    public static void main(String[] args) {
        InfoSunject infoSunject=new InfoSunject();

        Observer observer=new Observer1(infoSunject);

        infoSunject.setInfo("123");
        infoSunject.setInfo("246");
        infoSunject.setInfo("246");
    }


    public abstract class A{}

    public abstract  class B  extends A {}
}
