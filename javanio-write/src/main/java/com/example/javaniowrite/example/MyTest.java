package com.example.javaniowrite.example;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-07-29 20:43
 */
public class MyTest {
    private static final long count = 10000001;

    public static void main(String[] args) throws InterruptedException {
        concurrency();
        serial();
    }

    private static void concurrency() throws InterruptedException {
        long start = System.currentTimeMillis();
        Thread thread = new Thread(() -> {
            int a = 0;
            for (long i = 0; i < count; i++) {
                a += 5;
            }
        });

        thread.start(); //新线程加
        int b = 0;
        //原线程减
        for (long i = 0; i < count; i++) {
            b--;
        }
        thread.join();
        long time = System.currentTimeMillis() - start;
        System.out.println("concurrency : " + time + "ms, b = " + b);
    }

    public static void serial() {
        long start = System.currentTimeMillis();
        int a = 0;
        for(long i = 0; i < count; i++) {
            a += 5;
        }
        int b = 0;
        for (long i = 0; i < count; i++) {
            b--;
        }
        long time = System.currentTimeMillis() - start;
        System.out.println("serial:" + time + "ms, b = " + b + ", a = " + a);
    }
}
