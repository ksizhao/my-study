package com.example.javaniowrite.IO;

import java.io.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author zhaoliancan
 * @description NIO测试类
 * @create 2019-02-20 13:23
 */
public class NioTest {
    public static void main(String[] args) {

        Path file = Paths.get("E:\\mytest\\1.txt");
        AsynchronousFileChannel channel = null;
        try {
            channel = AsynchronousFileChannel.open(file);
            channel.read(ByteBuffer.allocate(100_000), 0, null, new CompletionHandler<Integer, ByteBuffer>() {
                @Override
                public void completed(Integer result, ByteBuffer attachment) {
                    System.out.println("读取结果"+result);
                    System.out.println("读取结果"+attachment.toString());
                }
                @Override
                public void failed(Throwable exc, ByteBuffer attachment) {
                    System.out.println(exc.getMessage());
                }
            });

            System.out.println("主线程继续做事情");
        } catch (IOException e) {
            e.printStackTrace();
        }


   //     writeNIO();


    }

    private static void readFile(File file) {
        RandomAccessFile raf=null;

        try {
            raf=new RandomAccessFile(file,"rw");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        FileChannel fileChannel=raf.getChannel();
        ByteBuffer buf=ByteBuffer.allocate(1024);
        System.out.println("位置"+buf.position()+"限制"+buf.limit()+"容量"+buf.capacity());
        int length=-1;
        try {
            while ((length=fileChannel.read(buf)) !=-1) {

                buf.clear();
                byte [] bytes=buf.array();
                String str=new String(bytes);
                System.out.println("读取内容"+str);
                System.out.println("位置"+buf.position()+"限制"+buf.limit()+"容量"+buf.capacity());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            raf.close();
            fileChannel.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static   void writeFile(File file) throws IOException {
        FileOutputStream fos=null;
        if (!file.exists()) {

            file.createNewFile();
        }
        fos=new FileOutputStream(file);
        FileChannel fileChannel=fos.getChannel();
        //ByteBuffer byteBuffer=Charset.forName("utf8").encode("你好AaxAX");
        ByteBuffer byteBuffer=ByteBuffer.allocate(1024);
        for (int i=0;i<100;i++) {
            byteBuffer.putInt(i);
        }
        System.out.println("缓冲区内容"+byteBuffer.get());
        //byteBuffer.put("123".getBytes());
        System.out.println("初始化容量：" + byteBuffer.capacity() + ",limit：" + byteBuffer.limit()+"posi:"+byteBuffer.position());
        int byteWrite=0;
        while ( (byteWrite=fileChannel.write(byteBuffer)) !=0) {
            System.out.println("写入长度"+byteWrite);
            System.out.println("写入内容"+byteBuffer.toString());
        }
        fos.close();
    }

    public static void writeNIO() {
               FileOutputStream fos = null;
                try {

                    fos = new FileOutputStream(new File("E:\\mytest\\mytest.txt"));
                         FileChannel channel = fos.getChannel();
                        ByteBuffer src = Charset.forName("utf8").encode("你好");
                        // 字节缓冲的容量和limit会随着数据长度变化，不是固定不变的
                        System.out.println("初始化容量和limit：" + src.capacity() + ","
                                         + src.limit());
                       int length = 0;

                       while ((length = channel.write(src)) != 0) {


                                System.out.println("写入长度:" + length);
                            }

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                    e.printStackTrace();
                    } finally {
                         if (fos != null) {
                                 try {
                                         fos.close();
                                     } catch (IOException e) {
                                         e.printStackTrace();
                                     }
                             }
                     }
             }

}
