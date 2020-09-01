package com.example.javaniowrite.thread;

import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author zhaoliancan
 * @description 倒计时线程
 * @create 2019-03-28 20:24
 */
public class CountThread implements Runnable {

    static final CountDownLatch count=new CountDownLatch(10);
    static CountThread countThread=new CountThread();

    @Override
    public void run() {

        try {
            Thread.sleep(1000L);
            System.out.println(Thread.currentThread().getName()+"执行结束");

        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            count.countDown();
        }

    }

    public static void main(String[] args) throws InterruptedException {
        ExecutorService executorService=Executors.newFixedThreadPool(10);
        for (int i=0; i<10000; i++) {
            executorService.submit(countThread);
        }
        count.await();
        System.out.println("主程序开始执行");
    }
}
