package com.example.javaniowrite.proxy;

import com.example.javaniowrite.simpleton.SimpleLon;

import java.io.FileReader;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.Properties;

/**
 * @author zhaoliancan
 * @description 测试类
 * @create 2019-01-03 19:21
 */
public class Test {

    public static void main(String[] args) throws Exception{

       /*  静态代理
        RealMovie realMovie=new RealMovie();
        Movie movie=new Cinema(realMovie);
        movie.play();
        */



        RealMovie realMovie=new RealMovie();
        ImaxMovie imax=new ImaxMovie();

        Maotai maotai=new Maotai();

        InvocationHandler movie1=new Imax(realMovie);
        InvocationHandler movie2=new Imax(imax);

        InvocationHandler wine=new Imax(maotai);

        Movie proxy= (Movie) Proxy.newProxyInstance(RealMovie.class.getClassLoader(),RealMovie.class.getInterfaces(),movie1);
        Movie proxy1= (Movie) Proxy.newProxyInstance(ImaxMovie.class.getClassLoader(),ImaxMovie.class.getInterfaces(),movie2);

        SellWine proxy2= (SellWine) Proxy.newProxyInstance(Maotai.class.getClassLoader(),Maotai.class.getInterfaces(),wine);
        Class class2=SimpleLon.class;
        try {
            Class class1=Class.forName("com.example.javaniowrite.simpleton.SimpleLon");
            System.out.println(class1==class2);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        proxy.play();
        proxy1.play();
        proxy2.sold();
        proxy2.showName();
        System.out.println("代理类名1"+proxy.getClass().getName());
        System.out.println("代理类名2"+proxy1.getClass().getName());
        System.out.println("代理类名3"+proxy2.getClass().getName());


       /* 加载配置内容
        try {
            Class className=Class.forName(getVelue("class"));
            Method m=className.getMethod(getVelue("className"));
            m.invoke(className.getConstructor().newInstance());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        */

//        ArrayList<String> stringList=new ArrayList<>();
//        stringList.add("zlc");
//        stringList.add("douible");
//
//        Class className=stringList.getClass();
//        Class class1=ArrayList.class;
//        Class class2=Class.forName("ArrayList");
//        className.getConstructors();// 获取方法
//        className.getFields();//获取属性
//        Method method=className.getMethod("add",Object.class);
//        method.invoke(stringList,100);
//        for (Object s:stringList) {
//            System.out.println("输出值："+s);
//        }


    }

    public  static String getVelue(String key) throws Exception{


        FileReader file=new FileReader("E:\\pro.txt");
        Properties properties=new Properties();
        properties.load(file);
        file.close();
        return properties.getProperty(key);

    }
}
