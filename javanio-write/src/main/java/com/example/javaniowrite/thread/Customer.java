package com.example.javaniowrite.thread;

import sun.misc.Unsafe;

/**
 * @author zhaoliancan
 * @description 消费类
 * @create 2019-03-19 20:40
 */
public class Customer {
    private int count=0;
    private int maxCount=100000;
    private String  LOCK="lock";


    class Producer implements Runnable {
        @Override
        public void run() {

            for (int i=0; i<100000; i++) {

                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (LOCK) {

                    while (count == maxCount) {
                        try {
                            LOCK.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    count++;
                    System.out.println(Thread.currentThread().getName()+"生产一个，,目前共"+count+"个");
                    LOCK.notifyAll();
                }
            }


        }
    }


    class Customers implements Runnable {
        @Override
        public void run() {
            for (int i=0; i<100000; i++) {

                try {
                    Thread.sleep(4000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (LOCK) {

                    while (count == 0) {
                        try {
                            LOCK.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    count--;
                    System.out.println(Thread.currentThread().getName()+"消费一个，目前共"+count+"个");
                    LOCK.notifyAll();
                }
            }
        }
    }

    public static void main(String[] args) {
        Customer customer=new Customer();
        new Thread(customer.new Producer()).start();
        new Thread(customer.new Customers()).start();
        new Thread(customer.new Producer()).start();
        new Thread(customer.new Producer()).start();
        Unsafe unsafe=Unsafe.getUnsafe();
        System.out.println(unsafe);

    }
}
