package com.example.javaniowrite.thread;

import java.io.*;
import java.util.Date;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 线程测试类
 * @create 2019-02-22 10:55
 */
public class MyThread  implements Runnable {


    private String commond;

    public MyThread(String commond) {
        this.commond=commond;
    }

    @Override
    public String toString() {
        return "MyThread{" +
                "commond='" + commond + '\'' +
                '}';
    }

    @Override
    public void run() {
        String name="E:\\mytest\\mytest.txt";
      //  readFile(name);
        System.out.println("当前线程："+Thread.currentThread().getName()+"开始时间"+(new Date()));
        operation();
        System.out.println("当前线程："+Thread.currentThread().getName()+"结束时间"+(new Date()));

    }

    private void operation() {

        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void readFile(String fileName) {

        File file=new File(fileName);
        AtomicInteger count=new AtomicInteger(0);
        long startTime=System.currentTimeMillis();
        if (file.exists()) {

            try {
                FileInputStream fis=new FileInputStream(file);
                byte[] bytes=new byte[1024];
                try {
                    while (fis.read(bytes) !=-1) {
                        System.out.println(new String(bytes));
                        count.getAndIncrement();
                        bytes=new byte[1024];

                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                long endtTime=System.currentTimeMillis();

                System.out.println("读文件耗时"+(endtTime-startTime)/1000+"秒"+(endtTime-startTime)%1000+"毫秒");


                try {
                    fis.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }


            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

        }else {
            System.out.println("该文件目录不存在");
        }
    }
}
