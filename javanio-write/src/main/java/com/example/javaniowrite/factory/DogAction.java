package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 狗的动作
 * @create 2019-02-12 19:45
 */
public class DogAction  implements Action{

    @Override
    public void behaviour() {
        System.out.println("我是狗，我会看门");
    }
}
