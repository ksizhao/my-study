package com.example.javaniowrite.thread;

import java.util.HashSet;
import java.util.Set;

/**
 * @author zhaoliancan
 * @description 线程测试类
 * @create 2019-03-25 19:24
 */
public class JoinTest {
    public volatile static int i=0;

   public static class JoinThread extends Thread {

        @Override
        public void run() {
            //super.run();
            while (true) {

                for (int j=0; j<100000; j++) {
                    i++;
                }
                System.out.println("2233");
            }
        }
    }


    public  static class High extends Thread {
       static  int count=0;

        @Override
        public void run() {
            //
            while (true) {
                synchronized (JoinTest.class) {
                    count++;
                    if (count >1000000) {
                        System.out.println("计数值"+count);
                        System.out.println("高优先级线程执行结束");
                        break;
                    }
                }
            }
        }
    }

    public  static class Low extends Thread {
        static  int count=0;

        @Override
        public void run() {
            //
            while (true) {
                synchronized (JoinTest.class) {
                    count++;
                    if (count >1000000) {
                        System.out.println("计数值"+count);
                        System.out.println("低优先级线程执行结束");
                        break;
                    }
                }
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
//        JoinThread joinThread=new JoinThread();
//        // 设置为守护线程
//        joinThread.setDaemon(true);
//        joinThread.start();
//        //joinThread.join();
//        System.out.println(i);
//
//        High high=new High();
//        Low low=new Low();
//        high.setPriority(6);
//        low.setPriority(5);
//        low.start();
//        high.start();
        int [] array={10,45,7,8,56,78,12,74,41};
        quickSort1(array,0,array.length-1);
        System.out.println(array.toString());

    }

    public boolean containsDuplicate(int[] nums) {
       Set<Integer> integerSet=new HashSet<>();
        for (int j = 0; j < nums.length; j++) {
           if (integerSet.contains(nums[j])) {
               return true;
           }
            integerSet.add(nums[j]);
        }
        return false;


    }

//
//    public static void quickSort(int [] num,int left,int right) {
//
//        if (left > right) {
//
//            return ;
//        }
//       int base=num[left];
//       int i=left;
//       int j=right;
//
//       while (i<j) {
//
//           while (base <= num[j] && i < j) {
//               j--;
//           }
//           while (base >= num[i] && i < j) {
//               i++;
//           }
//
//
//           if (i < j) {
//               int temp=num[i];
//               num[i]=num[j];
//               num[j]=temp;
//           }
//
//       }
//
//       num[left]=num[i];
//       num[i]=base;
//
//       quickSort(num,left,i-1);
//       quickSort(num,i+1,right);
//    }

    private static void quickSort1(int[] array, int left, int right) {

        if (left > right) {
            return;
        }
        int base = array[left];
        int i = left;
        int j = right;
        while (i < j) {
            while (base <= array[j] && i < j) {
                j--;
            }
            while (base >= array[i] && i < j) {
                i++;
            }

            if (i < j) {
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }

        }
        array[left] = array[i];
        // 把基准放到合适的位置
        array[i] = base;
        //对左边的子数组进行快速排序
        quickSort1(array, left, i - 1);
        //对右边的子数组进行快速排序
        quickSort1(array, i + 1, right);

    }
 }
