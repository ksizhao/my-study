package com.example.javaniowrite.designModule.adapter;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;

/**
 * @author zhaoliancan
 * @description 集合适配器
 * @create 2019-06-10 16:31
 */
public class ArrayListAdapter  implements Enumeration<ArrayList> {

    Iterator<ArrayList> listIterator;

    public ArrayListAdapter(Iterator<ArrayList> iterator) {
        listIterator=iterator;
    }


    @Override
    public boolean hasMoreElements() {
        return listIterator.hasNext();
    }

    @Override
    public ArrayList nextElement() {
        return listIterator.next();
    }
}
