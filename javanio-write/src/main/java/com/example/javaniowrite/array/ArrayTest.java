package com.example.javaniowrite.array;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年05月06日 11:01:00
 */
public class ArrayTest {
    static List<Integer> array = new ArrayList<Integer>();
    static List<Integer> linked = new LinkedList<Integer>();
    static int count=100000;


    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < count; i++) {
            array.add(i);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("array耗时："+(endTime-startTime));
        long startTime1 = System.currentTimeMillis();
        for (int i = 0; i < count; i++) {
            linked.add(i);
        }
        long endTime1 = System.currentTimeMillis();
        System.out.println("linked耗时："+(endTime1-startTime1));
    }
}
