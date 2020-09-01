package com.example.javaniowrite.designModule.command;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-06-05 20:33
 */
public class Test {

    public static void main(String[] args) {
        Light light=new Light();
        LightCommand lightCommand=new LightCommand(light);
        RemoteControl remoteControl=new RemoteControl(lightCommand);
        remoteControl.buttonPress();
    }
}
