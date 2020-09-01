package com.example.javaniowrite.proxy;

import jdk.nashorn.internal.ir.WhileNode;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 原子操作测试类
 * @create 2019-02-12 14:40
 */
public class ValoiteTest {

    private static volatile int count=0;

    private static AtomicInteger atomicInteger=new AtomicInteger(0);
    private static final int NUM=20;
    private static synchronized void increase() {
       // 非原子操作，加1，写值
        count++;
    }

    private static   void  increaseByAtomicInteger() {
        //原子操作
        atomicInteger.getAndIncrement();
    }

    public static void main(String[] args) {
        Thread [] threads=new Thread[NUM];
        long startTime=System.currentTimeMillis();
        for (int i=0;i<NUM;i++) {
            threads[i]=new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int i=0;i<100000;i++) {
                      // increase();
                        increaseByAtomicInteger();
                    }

                }
            });
            threads[i].start();
        }
        // 等待所有累加线程都结束
        while (Thread.activeCount() > 1) {
            Thread.yield();
        }
        long endTime=System.currentTimeMillis();
        System.out.println("耗时"+(endTime-startTime)+"毫秒");
        System.out.println("计数值"+atomicInteger);
    }

}
