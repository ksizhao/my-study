package com.example.javaniowrite.designModule.command;

/**
 * @author zhaoliancan
 * @description 调用者
 * @create 2019-06-05 20:22
 */
public class RemoteControl {
    Command command;

    public RemoteControl(Command command) {

        this.command=command;

    }

    public void buttonPress() {
        command.execute();
    }

}
