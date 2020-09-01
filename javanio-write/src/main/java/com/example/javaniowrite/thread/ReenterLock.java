package com.example.javaniowrite.thread;

import io.netty.util.concurrent.EventExecutor;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author zhaoliancan
 * @description 可重入锁
 * @create 2019-03-27 20:24
 */
public class ReenterLock implements Runnable {

    public static ReentrantLock r1=new ReentrantLock();
    public static Condition condition=r1.newCondition();

    @Override
    public void run() {

        try {
            r1.lock();
            condition.await();
            System.out.println("线程开始等待");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }finally {
            r1.unlock();
        }

    }

    public static void main(String[] args) throws InterruptedException {
        ReenterLock reenterLock=new ReenterLock();
        ExecutorService eventExecutor=Executors.newSingleThreadExecutor();
        Thread t1=new Thread(reenterLock);
        eventExecutor.execute(t1);
        Thread.sleep(2000L);
        r1.lock();
        condition.signal();
       // r1.unlock();
    }
}
