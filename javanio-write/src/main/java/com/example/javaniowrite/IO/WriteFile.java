package com.example.javaniowrite.IO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zhaoliancan
 * @description 多并发写文件
 * @create 2019-01-02 10:53
 */

public class WriteFile  extends Thread{

    private static final Logger logger = LoggerFactory.getLogger(WriteFile.class);
    private StringRedisTemplate redisTemplate;
    protected  static  final int NUM=200000;





//    @Autowired
//    private StringRedisTemplate redisTemplate;
//    @Qualifier("recordRedisTemplate")
//    @Autowired
//    private StringRedisTemplate recordRedisTemplate;
//
//    @Qualifier("serverRedisTemplete")
//    @Autowired
//    private StringRedisTemplate serverRedisTemplete;


    @Override
    public void run() {
        System.out.println("开始向文件写入数据");
        Calendar startTime=Calendar.getInstance();
        File writeFile=new File("E:\\mytest\\mytest.txt");
        try {
          if (!writeFile.exists()) {


              writeFile.createNewFile();
          }
          RandomAccessFile accessFile=new RandomAccessFile(writeFile,"rw");
          FileChannel fileChannel=accessFile.getChannel();
          FileLock fileLock=null;



          while (true) {
              try {
                  fileLock=fileChannel.lock();
                  break;

                  }catch (Exception e) {
                  e.printStackTrace();
                  logger.warn("有其他线程正在操作改文件,线程等待1000毫秒");
                  sleep(2000);

                  }
          }

          for (int i= 0; i<NUM; i++){
              StringBuffer sb=new StringBuffer();
              sb.append("这是第"+i+"行\r\n");
              String key="count:" + i;
//              try {
//              redisTemplate.opsForValue().set(key,"这是第"+i+"行");
//
//              }catch (Exception e) {
//                  e.printStackTrace();
//                  logger.warn("向redis写入数据失败，{}",e);
//
//              }

              accessFile.write(sb.toString().getBytes("UTF-8"));

          }

          fileLock.release();
          fileChannel.close();
          accessFile.close();
          accessFile=null;
          Calendar endTime=Calendar.getInstance();
          logger.info("本次写文件用时"+(endTime.getTimeInMillis()-startTime.getTimeInMillis())+"毫秒");

          }catch (IOException e){
            e.printStackTrace();
            logger.warn("写文件发生异常,异常为{}",e);

        }catch (InterruptedException e1) {
                e1.printStackTrace();
            }
        }



        public int [] getSum(int [] nums,int target) {

            HashMap<Integer,Integer> hashMap=new HashMap<>();
            for (int i=0;i<nums.length;i++) {
                if (hashMap.containsKey(target-nums[i])) {
                    return  new int [] { hashMap.get(target-nums[i]),i};
                }
                hashMap.put(nums[i],i);
            }
            
            return null;

        }







    }

