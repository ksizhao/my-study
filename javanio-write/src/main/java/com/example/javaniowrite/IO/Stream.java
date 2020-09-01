package com.example.javaniowrite.IO;

import javafx.collections.transformation.SortedList;

import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaoliancan
 * @description 流
 * @create 2019-02-25 18:59
 */
public class Stream {

    public static void main(String[] args) {

      getList();

    }


    public static void   getList(){

        List<String> stringList=new LinkedList<>();
        stringList.add("123456");
        stringList.add("12345742");
        stringList.add("17884");
        java.util.stream.Stream<String> stringStream=stringList.stream();
        long count=stringStream.filter(s -> s.length()>6).count();
        System.out.println("计数值"+count);
    }





}
