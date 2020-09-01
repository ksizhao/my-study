package com.example.javaniowrite.IO;

import java.io.*;
import java.nio.Buffer;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 文件操作类
 * @create 2019-02-14 10:28
 */
public class operationFile {

    public static void main(String[] args) {

        File file=new File("E:\\mytest\\test2.txt");
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
                int count1=getLineNumber(file);
                System.out.println("读文件耗时"+(endtTime-startTime)/1000+"秒"+(endtTime-startTime)%1000+"毫秒");
                System.out.println("共读取"+count1+"条数据");

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


    private static int getLineNumber(File fileName) {
        int count=0;
        LineNumberReader reader=null;

        if (fileName.exists()) {

            try {
                reader= new LineNumberReader(new InputStreamReader(new FileInputStream(fileName)));
                while (reader.readLine() !=null) {

                }
                count=reader.getLineNumber();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (reader !=null){

                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }

        return  count;
    }
}
