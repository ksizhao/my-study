package com.example.javaniowrite.example;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * @author zhaoliancan
 * @description 数组类
 * @create 2019-04-17 19:27
 */
public class ArrayTest {


    /**
     * 计算公共前缀
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {

            if ( strs == null || strs.length == 0) {
                return "";
            }
            String prefix=strs[0];
            for (int i = 1; i < strs.length; i++) {
                while (strs[i].indexOf(prefix) != 0) {
                    prefix=prefix.substring(0,prefix.length()-1);
                    if (prefix.isEmpty()) {
                        return "";
                    }
                }

            }
            return prefix;

    }

    /**
     * 求数组中只出现一次的数字
     * @param nums
     * @return
     */
    public static  int singleNumber(int[] nums) {
        if (nums == null) {
            return 0;
        }
        int size=nums.length;
        int result=0;
        for (int i=0; i<size; i++) {

            result =nums[i]^result;
        }
        return result;

    }

    /**
     * 数组求交集
     * @param nums1
     * @param nums2
     * @return
     */
    public static int[] intersect(int[] nums1, int[] nums2) {

        if (nums1 ==null || nums2 == null) {
            return null;
        }
        List<Integer> list1=new LinkedList();
        List<Integer> list2=new LinkedList();

        for (Integer n:nums1) {
            list1.add(n);
        }
        for (Integer num:nums2) {
            if (list1.contains(num)) {
                list2.add(num);
                // 移除已经匹配的
                list1.remove(num);
            }
        }
        int i=0;
        int [] array=new int [list2.size()];
        for (int j=0;j<list2.size(); j++) {
            array[i++]=list2.get(j);
        }
        return array;
    }


    /**
     * 字符串倒序排列
     * @param chars
     * @return
     */
    public  static  char[] reverse(char [] chars) {

        int i=0, j=chars.length-1;

        while (i < j) {
            char temp=chars[i];
            chars[i]=chars[j];
            chars[j]=temp;
            i++;
            j--;
        }
        System.out.println(chars);

        return  chars;
    }

    public static void merge(int[] nums1, int m, int[] nums2, int n) {

        int k=m;
        for (int i=0; i< nums1.length;i++) {
            if (k < nums1.length) {
                nums1[k]=nums2[i];
            }
            k++;
        }
        sort(nums1);
        System.out.println(nums1);
    }

    public static void sort(int [] array) {
        for (int i=0; i<array.length; i++) {
            for (int j=i+1; j<array.length; j++) {
                if (array[i] > array[j]) {
                    int tmp=array[j];
                    array[j]=array[i];
                    array[i]=tmp;
                }
             }
        }
    }

    public static void merge1(int[] nums1, int m, int[] nums2, int n) {
        int index = n+m-1;// 标记
        while (n>0&& m>0){
            if (nums1[m-1]<=nums2[n-1]){
                nums1[index] = nums2[n-1];
                n--;
            }else {
                nums1[index] = nums1[m-1];
                m--;
            }
            index--;
        }
        for (int i = 0;i<n;i++){
            nums1[i] = nums2[i];
        }
        System.out.println(nums1);

    }

    public void reverseString(char[] s) {
        int i=0, j=s.length-1;
        while (i < j) {
            char temp=s[i];
            s[i]=s[j];
            s[j]=temp;
            i++;
            j--;
        }

    }

    private static void print(int year) {
        Calendar c = Calendar.getInstance();
        System.out.println(year + "年");
        for(int i = 0 ; i < 12 ; i++){
            c.set(Calendar.MONTH, i);
            int month = c.get(Calendar.MONTH)+1;
            c.set(Calendar.DAY_OF_MONTH, 1);
            c.set(Calendar.YEAR, year);
            int date = c.get(Calendar.DAY_OF_WEEK);
            while(c.get(Calendar.MONTH) == month - 1){ //重点，推出循环
                if(c.get(Calendar.DAY_OF_MONTH) < 10){
                    if(c.get(Calendar.DAY_OF_WEEK) == 1 || c.get(Calendar.DAY_OF_WEEK) == 7){
                        System.out.println(""+year+"-"+(month<10?"0"+month:month)+"-0"+c.get(Calendar.DAY_OF_MONTH) + "周末");
                        //1
                    }else{
                        System.out.println(""+year+"-"+(month<10?"0"+month:month)+"-0"+c.get(Calendar.DAY_OF_MONTH));
                        //0
                    }
                }else{
                    if(c.get(Calendar.DAY_OF_WEEK) == 1 || c.get(Calendar.DAY_OF_WEEK) == 7){
                        System.out.println(""+year+"-"+(month<10?"0"+month:month)+"-"+c.get(Calendar.DAY_OF_MONTH) + "周末");
                        //1
                    }else{
                        System.out.println(""+year+"-"+(month<10?"0"+month:month)+"-"+c.get(Calendar.DAY_OF_MONTH));
                        //0
                    }
                }
                date = c.get(Calendar.DAY_OF_MONTH);
                date = date + 1;
                c.set(Calendar.DAY_OF_MONTH, date);
            }
        }
    }




        public static void main(String[] args) {
            int year = 2019;
            print(year);


       // System.out.println("结果"+result);




    }

}
