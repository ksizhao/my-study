package com.example.javaniowrite.designModule.state.impl;

import com.example.javaniowrite.designModule.state.EntranceMachine;
import com.example.javaniowrite.designModule.state.EntranceMachineState;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年05月19日 11:34:00
 */
public class LockedEntranceMachineState implements EntranceMachineState {

    @Override
    public String insertCoin(EntranceMachine entranceMachine) {
        return entranceMachine.open();
    }

    @Override
    public String pass(EntranceMachine entranceMachine) {
        return entranceMachine.alarm();
    }
}
