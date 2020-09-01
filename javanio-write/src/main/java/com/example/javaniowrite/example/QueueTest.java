package com.example.javaniowrite.example;

import io.netty.util.concurrent.DefaultThreadFactory;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * @author zhaoliancan
 * @description 队列测试类
 * @create 2019-04-24 19:49
 */
public class QueueTest {

    private static ConcurrentLinkedQueue linkedQueue=new ConcurrentLinkedQueue();
    private static ConcurrentSkipListMap<String,Integer> map=new ConcurrentSkipListMap<>();
    private static List<Integer> integerList=Collections.synchronizedList(new LinkedList<>());

    static class MyThread implements Runnable{

        @Override
        public void run() {

            for (int i=0; i<100000; i++) {
                //linkedQueue.add(i);
                map.put(String.valueOf(i),i);
            }
        }
    }

    public static void main(String[] args) {
        ExecutorService executorService=Executors.newCachedThreadPool();
        long start=System.currentTimeMillis();
        for (int i=0; i<50; i++) {
            Runnable t1=new MyThread();
            executorService.execute(t1);
        }
        executorService.shutdown();
//        long end=System.currentTimeMillis();
//        long time=end-start;
//        System.out.println("大小"+linkedQueue.size());
//        System.out.println("耗时"+time+"毫秒");
        for (Map.Entry<String,Integer> entry:map.entrySet()) {
            System.out.println("键"+entry.getKey());
        }
        System.exit(0);
    }

}
