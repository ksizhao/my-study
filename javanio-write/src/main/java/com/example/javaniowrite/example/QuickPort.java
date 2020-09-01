package com.example.javaniowrite.example;

import java.util.Arrays;

/**
 * @author zhaoliancan
 * @description 排序算法
 * @create 2019-02-27 20:25
 */
public class QuickPort {

    public static void main(String[] args) {

        int [] array={10,45,7,8,56,78,12,74,41};
        //int [] array=randomCommon(1,1000,5000);
//          quickSort(array,0,array.length-1);
//          System.out.println(array);
//        long startTime=System.currentTimeMillis();
//        quickSort3(array,0,array.length-1);
//        long endTime=System.currentTimeMillis();
//        System.out.println("快速排序耗时"+(endTime-startTime));
//        long startTime1=System.currentTimeMillis();
//        maopaoSort(array);
//        long endTime1=System.currentTimeMillis();
//        System.out.println("冒泡排序耗时"+(endTime1-startTime1));
//        int [] result=maopaoSort(array);
//        System.out.println(result);

    }


    /**
     * 插入排序
     * @param array
     * @return
     */
    public static int [] insertSort(int [] array) {

        int [] newArray=Arrays.copyOf(array,array.length);

        for (int i=1; i<newArray.length; i++) {
            // 当前要插入的值
            int tmp=newArray[i];
            int j=i;
            // 从已排序的最右边开始比较，找到比其小的值
            while (j > 0 && tmp < newArray[j-1]) {
                newArray[j]=newArray[j-1];
                j--;
            }

            //找到比其小的值，则插入
            if (j != i) {
                newArray[j]=tmp;
            }
        }

        return newArray;

    }

    public static int[] randomCommon(int min, int max, int n){
        if (n > (max - min + 1) || max < min) {
            return null;
        }
        int[] result = new int[n];
        int count = 0;
        while(count < n) {
            int num = (int) (Math.random() * (max - min)) + min;
            boolean flag = true;
            for (int j = 0; j < n; j++) {
                if(num == result[j]){
                    flag = false;
                    break;
                }
            }
            if(flag){
                result[count] = num;
                count++;
            }
        }
        return result;
    }

//    private static void quickSort(int[] array, int left, int right) {
//
//        if (left > right) {
//            return;
//        }
//        int base = array[left];
//        int i = left;
//        int j = right;
//        while (i < j) {
//            while (base <= array[j] && i < j) {
//                j--;
//            }
//            while (base >= array[i] && i < j) {
//                i++;
//            }
//
//            if (i < j) {
//                int temp = array[i];
//                array[i] = array[j];
//                array[j] = temp;
//            }
//
//        }
//        array[left] = array[i];
//        // 把基准放到合适的位置
//        array[i] = base;
//
//        //对左边的子数组进行快速排序
//        quickSort(array, left, i - 1);
//        //对右边的子数组进行快速排序
//        quickSort(array, i + 1, right);
//
//
//    }

//    public static void quickSort1(int[]a,int left,int right)
//    {
//        if(left>right)
//            return;
//        int pivot=a[left];//定义基准值为数组第一个数
//        int i=left;
//        int j=right;
//
//        while(i<j)
//        {
//            while(pivot<=a[j]&&i<j)//从右往左找比基准值小的数
//                j--;
//            while(pivot>=a[i]&&i<j)//从左往右找比基准值大的数
//                i++;
//            if(i<j)                     //如果i<j，交换它们
//            {
//                int temp=a[i];
//                a[i]=a[j];
//                a[j]=temp;
//            }
//        }
//        a[left]=a[i];
//        a[i]=pivot;//把基准值放到合适的位置
//        quickSort1(a,left,i-1);//对左边的子数组进行快速排序
//        quickSort1(a,i+1,right);//对右边的子数组进行快速排序
//
//    }

    public static  int [] maopaoSort(int [] array) {
        for (int i=0;i<array.length;i++) {
            for (int j=i+1;j<array.length;j++){
                if (array[i] > array[j]) {
                    int temp=array[i];
                    array[i]=array[j];
                    array[j]=temp;

                }
            }
        }
        return array;
    }


}
