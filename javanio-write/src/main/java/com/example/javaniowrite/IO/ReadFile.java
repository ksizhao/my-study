package com.example.javaniowrite.IO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.Calendar;

/**
 * @author zhaoliancan
 * @description 多并发读文件
 * @create 2019-01-02 11:01
 */
public class ReadFile extends Thread{
    private static final Logger logger = LoggerFactory.getLogger(ReadFile.class);

    @Qualifier("recordRedisTemplate")
    @Autowired
    private StringRedisTemplate redisTemplate;

    @Override
    public void run() {
        //super.run();
        logger.info("开始从文件读取数据");
        Calendar startTime=Calendar.getInstance();
        File file=new File("E:\\mytest\\test.txt");
        try {
            RandomAccessFile accessFile=new RandomAccessFile(file,"rw");
            FileChannel fileChannel=accessFile.getChannel();
            FileLock fileLock=null;
            while (true) {
                try {
                    fileLock=fileChannel.tryLock();
                    break;
                }catch (Exception e) {
                    e.printStackTrace();
                    logger.info("当前文件被其他线程操作，睡眠1000毫秒",e);
                    sleep(1000L);

                }

            }

            byte [] bytes=new byte[1024];
            StringBuffer stringBuffer=new StringBuffer();
            while (accessFile.read(bytes) !=-1) {
                stringBuffer.append(bytes);
                bytes=new byte[1024];
            }

            fileLock.release();
            fileChannel.close();
            accessFile.close();
            accessFile=null;

            Calendar calendar=Calendar.getInstance();

            System.out.println("文件读取内容："+stringBuffer);
            System.out.println("本次读文件用时"+(calendar.getTimeInMillis()-startTime.getTimeInMillis())+"毫秒");

        }catch (Exception e){
            e.printStackTrace();
            logger.warn("读文件线程异常,异常为{}",e);
        }

    }
}
