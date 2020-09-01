package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 猫的动作
 * @create 2019-02-12 19:44
 */
public class CutAction implements Action {

    @Override
    public void behaviour() {
        System.out.println("我是猫，我会捉老鼠");
    }
}
