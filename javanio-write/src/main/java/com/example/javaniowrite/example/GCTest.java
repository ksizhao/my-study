package com.example.javaniowrite.example;

import org.springframework.util.StringUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author zhaoliancan
 * @description 垃圾测试类
 * @create 2019-03-04 20:34
 */
public class GCTest {

    public static void main(String[] args) {
//        byte [] bytes=new byte[100000*1024];
//      //  byte [] bytes1=new byte[100000*1024];
//        Path path= Paths.get("E:\\mytest\\test2.txt");
//        boolean flag=Files.exists(path);
//        if (!flag) {
//            System.out.println("该文件不存在");
//        }
//
//        try {
//            long startTime=System.currentTimeMillis();
//            List<String> stringList=Files.readAllLines(path,Charset.forName("UTF-8"));
//           // System.out.println("读取内容"+stringList);
//            Stream<String> stringStream=stringList.stream();
//            long count=stringStream.filter(s -> s.length()>1).count();
//            System.out.println("计数值"+count);
//            long endTime=System.currentTimeMillis();
//            System.out.println("耗时"+(endTime-startTime)/1000+"秒"+(endTime-startTime)%1000+"毫秒");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        String  [ ] strings={"flower","clow","dight"};
        String result= longestCommonPrefix(strings);
        System.out.println("最长公共前缀"+result);



    }


    public static String longestCommonPrefix(String[] strs) {

        if(strs ==null || strs.length == 0) {
            return "";
        }
        String prefix=strs[0];
        for (int i=0; i<strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix=prefix.substring(0,prefix.length()-1);
                if (StringUtils.isEmpty(prefix)) {
                    return "";
                }
            }
        }

        return prefix;
    }


}
