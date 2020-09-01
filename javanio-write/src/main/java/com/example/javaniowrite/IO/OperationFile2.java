package com.example.javaniowrite.IO;

import java.io.*;
import java.util.Calendar;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 写文件
 * @create 2019-02-20 9:38
 */
public class OperationFile2 {
    public static void main(String[] args) {


        File file=new File("E:\\mytest\\test2.txt");
        AtomicInteger count=new AtomicInteger(0);
        if ( !file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        try {
            FileOutputStream fos=new FileOutputStream(file);
            long startTime=System.currentTimeMillis();
            byte [] bytes=new byte[1024];
            for (int i=1; i<=2000000; i++) {
                try {
                    String str="这是第"+i+"行\r\n";
                    bytes=str.getBytes();
                    BufferedOutputStream bos=new BufferedOutputStream(fos);
                    bos.write(bytes);
                    bos.flush();

                 //   fos.write(bytes);
                    count.getAndIncrement();
                    bytes=new byte[1024];
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
            long endtTime=System.currentTimeMillis();
            System.out.println("写文件耗时"+(endtTime-startTime)/1000+"秒"+(endtTime-startTime)%1000+"毫秒");
            System.out.println("共写入"+count+"条数据");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }
}
