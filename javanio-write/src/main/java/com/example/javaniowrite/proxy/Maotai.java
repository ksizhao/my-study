package com.example.javaniowrite.proxy;

import java.util.*;

/**
 * @author zhaoliancan
 * @description 茅台酒
 * @create 2019-01-04 14:28
 */
public class Maotai implements SellWine {

    @Override
    public void sold() {
        System.out.println("我卖茅台酒");
    }

    @Override
    public void showName() {

        List<String> stringList=new LinkedList<>();
        stringList.add("123");
        Iterator<String> iterator=stringList.listIterator();
        while (iterator.hasNext()) {
            String string= iterator.next();
            System.out.println(string);
            ((ListIterator<String>) iterator).set("nima");
            System.out.println(string);
            iterator.remove();

        }

//
//        Map<Integer,String> map=new HashMap<>(100);
//        for ( int i=0; i<100; i++) {
//            map.put(i,"value"+i);
//
//        }
//
//        for (Map.Entry entry:map.entrySet()) {
//           System.out.println("读取值"+entry.getValue());
//        }
    }
}
