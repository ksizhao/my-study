package com.example.javaniowrite.example;

import org.springframework.cglib.proxy.Enhancer;
import org.springframework.context.annotation.Primary;
import sun.misc.Unsafe;

import java.io.FileDescriptor;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * @author zhaoliancan
 * @description 内存溢出
 * @create 2019-07-03 16:12
 */
public class MemoryOOM {
    public Object instance=null;
    static class OOM{
        public byte [] holder=new byte[64*1024];
    }

//    private static int SIZE = 1024*1014;
    public static void main(String[] args) {

        List<OOM> list=new LinkedList<>();
        for (int i = 0; i < 1000; i++) {
            try {
                Thread.sleep(200L);
                list.add(new OOM());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }
        System.gc();
//        Byte [] newSize1,newSize2,newSize3,newSize4;
//        newSize1 = new Byte[SIZE/4];
//        newSize2 = new Byte[1*SIZE];
//        newSize3 = new Byte[2*SIZE];
//        newSize4 = new Byte[1*SIZE];
       // ThreadPoolExecutor
//        Executors.newFixedThreadPool(100);
//        MemoryOOM a=new MemoryOOM();
//        MemoryOOM b=new MemoryOOM();
//        a.instance=b;
//        b.instance=a;
//        a=null;
//        b=null;
//        System.gc();
//        Field field=Unsafe.class.getDeclaredFields()[0];
//        field.setAccessible(true);
//        try {
//            Unsafe unsafe= (Unsafe) field.get(null);
//            while (true) {
//                unsafe.allocateMemory(SIZE);
//            }
//        } catch (IllegalAccessException e) {
//            e.printStackTrace();
//        }
//        HashMap<Integer, String> map = new HashMap(16);
//        map.put(7, "");
//        map.put(11, "");
//        map.put(43, "");
//        map.put(59, "");
//        map.put(19, "");
//        map.put(3, "");
//        map.put(35, "");
//
//        System.out.println("遍历结果：");
//        for (Integer key : map.keySet()) {
//            System.out.print(key + " -> ");
//        }
//        List<String> stringList = new ArrayList<>();
//        int i = 0;
//
//        Enhancer enhancer=new Enhancer();
//
//        while (true) {
//            stringList.add(String.valueOf(i++).intern());
//        }
    }
}
