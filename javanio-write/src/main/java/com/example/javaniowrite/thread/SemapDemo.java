package com.example.javaniowrite.thread;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

/**
 * @author zhaoliancan
 * @description 信号量
 * @create 2019-03-27 20:40
 */
public class SemapDemo implements Runnable {

    final Semaphore semaphore=new Semaphore(5);
    @Override
    public void run() {
        try {
            semaphore.acquire();
            Thread.sleep(1000L);
            System.out.println("当前线程"+Thread.currentThread().getId()+"在工作");
            semaphore.release();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        ExecutorService service=Executors.newFixedThreadPool(100);
        SemapDemo semapDemo=new SemapDemo();
        for (int i=0; i<100; i++) {
            service.execute(semapDemo);
        }
    }


}
