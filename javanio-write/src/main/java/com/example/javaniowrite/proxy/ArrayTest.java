package com.example.javaniowrite.proxy;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaoliancan
 * @description 数组测试类
 * @create 2019-01-23 17:06
 */
public class ArrayTest {
    public static void main(String[] args) {
        List<String> list =new ArrayList<>();
        Long startTime=System.currentTimeMillis();
        for (int i=0;i<100000;i++){
        list.add("value"+i);
        }
        Long endTime=System.currentTimeMillis();
        System.out.println("耗时："+(endTime-startTime));
        List<String> list1 =new ArrayList<>();
        Long startTime1=System.currentTimeMillis();
        ((ArrayList<String>) list1).ensureCapacity(100000);
        for (int i=0;i<100000;i++){
            list1.add("value"+i);
        }
        Long endTime1=System.currentTimeMillis();
        System.out.println("耗时1："+(endTime1-startTime1));

    }
}
