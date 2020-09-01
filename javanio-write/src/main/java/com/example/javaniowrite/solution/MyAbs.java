package com.example.javaniowrite.solution;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2020年06月04日 20:03:00
 */
abstract class MyAbs {

    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void tell(){
        System.out.println("test");
    }
}
