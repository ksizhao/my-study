package com.example.javaniowrite.thread;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年03月18日 17:38:00
 */
public class MyAtomicIntegerTest {
    private static final int THREADS_CONUT = 10;
    private static AtomicInteger count = new AtomicInteger(0);

    public static void increase() {
        count.getAndIncrement();
    }

    public static void main(String[] args) {
        Thread[] threads = new Thread[THREADS_CONUT];
        for (int i = 0; i < THREADS_CONUT; i++) {
            threads[i] = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int i = 0; i < 10000; i++) {
                        increase();
                    }
                }
            });
            threads[i].start();
        }

        while (Thread.activeCount() > 1) {
            Thread.yield();
        }
        System.out.println(count);
    }

}
