package com.example.javaniowrite.thread;

/**
 * @author zhaoliancan
 * @description 等待线程
 * @create 2019-03-14 20:38
 */
public class WaitThread {

    final static  Object OBJECT=new Object();
    public static class Thread1 extends Thread {

        @Override
        public void run() {
            //super.run();
            synchronized (OBJECT) {

                System.out.println(System.currentTimeMillis()+"t1 start");
                try {
                    System.out.println(System.currentTimeMillis()+"t1 to wait");
                    OBJECT.wait();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                System.out.println(System.currentTimeMillis()+"t1 end");
            }


        }
    }


    public static class Thread2 extends Thread {

        @Override
        public void run() {
            //super.run();
            synchronized (OBJECT) {

                System.out.println(System.currentTimeMillis()+"t2 start, notify one Object");
                OBJECT.notify();
                System.out.println(System.currentTimeMillis()+"t2 end");
                try {
                    Thread.sleep(2000L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

        }
    }

    public static void main(String[] args) {
        Thread1 thread1=new Thread1();
        Thread2 thread2=new Thread2();
        thread1.start();
        thread2.start();
    }
}
