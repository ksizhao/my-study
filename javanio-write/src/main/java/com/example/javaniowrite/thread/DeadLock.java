package com.example.javaniowrite.thread;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaoliancan
 * @description 死锁
 * @create 2019-04-18 20:42
 */
public class DeadLock implements Runnable {

    protected Object tool;
    static Object fork1=new Object();
    static Object fork2=new Object();
    static List<String> strings=Collections.synchronizedList(new LinkedList<>());

    public DeadLock (Object obj){
        this.tool=obj;
//        if (tool==fork1) {
//
//        }

    }



    @Override
    public void run() {
        if (tool == fork1) {
            synchronized (fork1) {
                try {
                    Thread.sleep(500L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (fork2) {
                    System.out.println("哲学家A开始吃饭了");
                }
            }
        }

        if (tool == fork2) {
            synchronized (fork2) {
                try {
                    Thread.sleep(500L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (fork1) {
                    System.out.println("哲学家B开始吃饭了");
                }
            }
        }

    }

    public static void main(String[] args) {
//        DeadLock a=new DeadLock(fork1);
//        DeadLock b=new DeadLock(fork2);
//        Thread thread=new Thread(a);
//        Thread thread1=new Thread(b);
//        thread.start();
//        thread1.start();
        for (;;){
            System.out.println("aaaa");
        }

    }
}
