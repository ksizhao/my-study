package com.example.javaniowrite.thread;

import sun.nio.ch.DirectBuffer;

import javax.annotation.PostConstruct;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 线程测试类
 * @create 2019-02-22 10:59
 */
public class ThreadTest {

    private static AtomicInteger count=new AtomicInteger(0);

    private final ExecutorService executorService;


    public ThreadTest(ExecutorService executorService) {
        this.executorService = executorService;
    }

    @PostConstruct
    public void init(){
        List<String> strings=new LinkedList<>();
        for (int j = 0; j < 10000; j++) {
            strings.add("test"+i);
        }
        CompletionService<String> completionServic=new ExecutorCompletionService<String>(executorService);
        for (String str:strings) {
            completionServic.submit(new Callable<String>() {
                @Override
                public String call() throws Exception {
                    return str+str;
                }
            });
        }

        for (int j = 0; j < strings.size(); j++) {
            try {
                Future<String> future=completionServic.take();
                System.out.println(future.get());
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    private volatile static int i=0;

    public static void main(String[] args) throws InterruptedException {

        ExecutorService executor=new ThreadPoolExecutor(500, 5000, 100L, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>(1000), new ThreadFactory() {



            @Override
            public Thread newThread(Runnable r) {
                Thread t=new Thread(r);
                // 设置为守护线程
                t.setDaemon(true);
                System.out.println("创建线程"+t.getName());
                return t;
            }
        }, new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {

                System.out.println(r.toString() + "被丢弃");

            }
        });
        for (int i=0;i<100;i++) {


            Runnable thread=new Threads();
           // executorService.schedule(thread,10,TimeUnit.SECONDS);
            // 每5秒执行一次
           // executorService.scheduleAtFixedRate(thread,5,10,TimeUnit.SECONDS);
         //   executorService.scheduleWithFixedDelay(thread,10,20,TimeUnit.SECONDS);
            executor.execute(thread);
            count.getAndIncrement();
        }
       // executorService.shutdown();
        System.out.println("共有"+count+"个线程执行");
        while (!executor.isTerminated()) {

        }
        executor.shutdown();


//        Thtrads thtrads=new Thtrads();
//        Thread t1=new Thread(thtrads);
//        Thread t2=new Thread(thtrads);
//        t1.start();
//        t2.start();
//        t1.join();
//        t2.join();
//        System.out.println(i);

    }

    private class QueueingFuture<V> extends FutureTask{

        public QueueingFuture(Callable<V> callable) {
            super(callable);
        }

        public QueueingFuture(Runnable runnable, Object result) {
            super(runnable, result);
        }
    }


    static  class Threads implements Runnable{

        @Override
        public void run() {
            for(int j=0; j<1000000; j++) {
                increase();
            }
            System.out.println("ThreadID:"+Thread.currentThread().getName()+System.currentTimeMillis());

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }


        public synchronized void increase() {
            i++;
        }
    }
}
