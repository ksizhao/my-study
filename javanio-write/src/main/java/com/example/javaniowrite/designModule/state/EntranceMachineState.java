package com.example.javaniowrite.designModule.state;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年05月19日 11:31:00
 */
public interface EntranceMachineState {

    /**
     * 投币
     * @param entranceMachine
     * @return
     */
    String insertCoin(EntranceMachine entranceMachine);

    /**
     * 通过
     * @param entranceMachine
     * @return
     */
    String pass(EntranceMachine entranceMachine);

}
