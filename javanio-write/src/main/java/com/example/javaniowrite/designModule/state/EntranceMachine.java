package com.example.javaniowrite.designModule.state;

import com.example.javaniowrite.designModule.state.impl.LockedEntranceMachineState;
import com.example.javaniowrite.designModule.state.impl.UnlockedEntranceMachineState;
import lombok.Data;

import java.util.Objects;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年05月19日 11:32:00
 */
@Data
public class EntranceMachine {
    private EntranceMachineState locked = new LockedEntranceMachineState();

    private EntranceMachineState unlocked = new UnlockedEntranceMachineState();

    private EntranceMachineState state;

    public EntranceMachine(EntranceMachineState state) {
        this.state = state;
    }

    public String execute(Action action) {
        if (Objects.isNull(action)) {
            throw new InvalidActionException();
        }

        if (Action.PASS.equals(action)) {
            return state.pass(this);
        }

        return state.insertCoin(this);
    }

    public boolean isUnlocked() {
        return state == unlocked;
    }

    public boolean isLocked() {
        return state == locked;
    }

    public String open() {
        setState(unlocked);
        return "opened";
    }

    public String alarm() {
        setState(locked);
        return "alarm";
    }

    public String refund() {
        setState(unlocked);
        return "refund";
    }

    public String close() {
        setState(locked);
        return "closed";
    }

    private void setState(EntranceMachineState state) {
        this.state = state;
    }
}
