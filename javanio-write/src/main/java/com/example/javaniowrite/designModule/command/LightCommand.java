package com.example.javaniowrite.designModule.command;

/**
 * @author zhaoliancan
 * @description 灯命令对象
 * @create 2019-06-05 20:24
 */
public class LightCommand implements Command {
    Light light;


    public LightCommand(Light light) {
        this.light=light;
    }

    @Override
    public void execute() {

        System.out.println("命令控制电灯");
        light.lightOn();
    }
}
