package com.example.javaniowrite.example;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-07-03 14:31
 */
public class OOMObject {

    static  class OMObject {

    }

    public static void main(String[] args) {
        List<OMObject> omObjects = new ArrayList<>();
        while (true) {
            omObjects.add(new OMObject());
        }
    }
}
