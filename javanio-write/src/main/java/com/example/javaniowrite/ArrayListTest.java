package com.example.javaniowrite;

import java.util.*;

/**
 * @author zhaoliancan
 * @description List测试类
 * @create 2019-02-13 10:14
 */
public class ArrayListTest {

    public static void main(String[] args) {
//        ArrayList<Integer> arrayList=new ArrayList<>();
//        arrayList.add(1);
//        arrayList.add(-8);
//        arrayList.add(5);
//        arrayList.add(3);
//        arrayList.add(10);
//        arrayList.add(20);
//
//        int size=arrayList.size();
//        Integer[] array=arrayList.toArray(new Integer[size]);
//        System.out.println("List转数组"+array[1]);
//
//
//
//        List<Integer> integers=new ArrayList<>();
//        integers.addAll(Arrays.asList(array));
//        System.out.println("list转ArrayList"+integers);
//
//        System.out.println("初始数组"+arrayList);
//        Collections.sort(arrayList);
//        System.out.println("自然升序排序后数组"+arrayList);
//        Collections.reverse(arrayList);
//        System.out.println("反转后数组"+arrayList);
//        arrayList.sort(new Comparator<Integer>() {
//            @Override
//            public int compare(Integer o1, Integer o2) {
//                return o2.compareTo(o1);
//            }
//        });
//        System.out.println("定制降序排序后数组"+arrayList);
        ThreadLocal<String> threadLocal=new ThreadLocal<>();
        threadLocal.set("A");
        String s=threadLocal.get();
        System.out.println(s);
//        Map<String, String> map = new HashMap<>();
//        map.put("one", "wang");
//        List<String> stringList = new LinkedList<>();
//        stringList.add("zlc");
//        Iterator<String> stringIterator = stringList.iterator();
//        while (stringIterator.hasNext()) {
//            String s = stringIterator.next();
//            System.out.println(s);
//        }
//        Set<Map.Entry<String, String>> entries = map.entrySet();
//        for (Map.Entry entry : entries) {
//            System.out.println("键:" + entry.getKey() + "值" + entry.getValue());
//        }

    }


}
