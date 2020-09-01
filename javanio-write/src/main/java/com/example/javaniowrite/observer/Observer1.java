package com.example.javaniowrite.observer;

import java.util.Observable;

/**
 * @author zhaoliancan
 * @description 观察者
 * @create 2019-05-23 13:29
 */
public class Observer1 implements java.util.Observer {

    Observable observable;
    private String info;

    /**
     * 进行更新操作
     * @param o
     * @param arg
     */
    @Override
    public void update(Observable o, Object arg) {

        if (o instanceof InfoSunject) {
            InfoSunject infoSunject= (InfoSunject) o;
             this.info=infoSunject.getInfo();
             display();
        }

    }

    /**
     * 注册观察者
     * @param observable
     */
    public Observer1(Observable observable) {
        this.observable=observable;
        observable.addObserver(this);
    }

    public void display() {
        System.out.println("输出信息"+info) ;
    }

}
