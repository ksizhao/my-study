package com.example.javaniowrite.observer;

import java.util.Observable;
import java.util.Observer;

/**
 * @author zhaoliancan
 * @description 主题对象
 * @create 2019-05-23 13:26
 */
public class InfoSunject extends Observable {

    private String info;

    public InfoSunject() {}


    /**
     * 通知观察者有了变化
     */
    public void  getInfoChange() {
        setChanged();
        notifyObservers();
    }

    public void setInfo(String info) {
        this.info=info;
        getInfoChange();
    }

    public String getInfo() {
        return info;
    }
}
