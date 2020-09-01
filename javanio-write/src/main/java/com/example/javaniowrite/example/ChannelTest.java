package com.example.javaniowrite.example;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Pipe;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;

/**
 * @author zhaoliancan
 * @description 文件管道
 * @create 2019-03-05 11:07
 */
public class ChannelTest {

    public static void main(String[] args) {
        try {
            //String path="";
//            File file=new File()
            Path path= Paths.get("E:\\mytest\\test2.txt");
            boolean flag=Files.exists(path);
            if (!flag) {
                System.out.println("该文件不存在");
            }
            List<String> stringList=Files.readAllLines(path,Charset.forName("UTF-8"));
            Pipe pipe=Pipe.open();
            Pipe.SinkChannel sinkChannel=pipe.sink();

            String str="这是我的测试";
            ByteBuffer byteBuffer=ByteBuffer.allocate(1024*100000);
            byteBuffer.clear();
            Iterator<String> stringIterator=stringList.listIterator();
            while (stringIterator.hasNext()) {
                byteBuffer.put(stringIterator.next().getBytes());
            }

            // 反转缓存区，准备读取
            byteBuffer.flip();
            while (byteBuffer.hasRemaining()) {
                sinkChannel.write(byteBuffer);
            }

            //功能反转，准备读取


            //源管道读取数据
            Pipe.SourceChannel sourceChannel=pipe.source();

            ByteBuffer   byteBuffer1=ByteBuffer.allocate(1024);

            //返回值代表读取数据量
            int count=sourceChannel.read(byteBuffer1);

            System.out.println("读取值"+count);

//            System.out.println("输出值"+new String(byteBuffer1.array()));



        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
