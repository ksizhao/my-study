package com.example.javaniowrite.example;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-05-31 10:43
 */
public class Solution {


    /**
     * 首先，我们设定一个哨兵节点 "prehead" ，这可以在最后让我们比较容易地返回合并后的链表。我们维护一个 prev 指针，我们需要做的是调整它的 next 指针。然后，我们重复以下过程，直到 l1 或者 l2 指向了 null ：如果 l1 当前位置的值小于等于 l2 ，我们就把 l1 的值接在 prev 节点的后面同时将 l1 指针往后移一个。否则，我们对 l2 做同样的操作。不管我们将哪一个元素接在了后面，我们都把 prev 向后移一个元素。
     * <p>
     * 在循环终止的时候， l1 和 l2 至多有一个是非空的。由于输入的两个链表都是有序的，所以不管哪个链表是非空的，它包含的所有元素都比前面已经合并链表中的所有元素都要大。这意味着我们只需要简单地将非空链表接在合并链表的后面，并返回合并链表。
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        ListNode preHead = new ListNode(-1);
        ListNode pre = preHead;
        while (l1 != null && l2 != null) {

            if (l1.val <= l2.val) {
                pre.next = l1;
                l1 = l1.next;
            } else {
                pre.next = l2;
                l2 = l2.next;
            }
            pre = pre.next;
        }

        //
        pre.next = l1 == null ? l2 : l1;

        return preHead.next;

    }


    private class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }


    /**
     * 动态规划
     *
     * @param n
     * @return
     */
    public static int climbStairs(int n) {

        if (n == 1) {
            return 1;
        }
        int array[] = new int[n + 1];
        array[1] = 1;
        array[2] = 2;
        for (int i = 3; i <= n; i++) {
            array[i] = array[i - 1] + array[i - 2];
        }

        return array[n];

    }

    /**
     * 斐波那契数
     *
     * @param n
     * @return
     */
    public static int climb(int n) {
        if (n == 1) {
            return 1;
        }
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = second + first;
            first = second;
            second = third;
        }
        return second;
    }

//    public static String reverseWords(String s) {
//
//        String [] strings = s.split(" ");
//        StringBuilder sb=new StringBuilder();
//        for (int i=0; i<strings.length; i++){
//            char [] chars=reverseString(strings[i]);
//            sb.append(chars);
//
//        }
//       return sb.toString();
//
//
//    }


//    public static String reverseWords(String s) {
//
//        int start=0;
//        int end=0;
//        StringBuilder sb=new StringBuilder();
//       for (int i = 0; i <= s.length(); i++) {
//           // 遇到空格符或结束符，反转前面字符串
//           if (s.charAt(i) == '\0' || s.charAt(i) == ' ') {
//               end=i;
//               String s1=s.substring(start,end);
//               char [] chars=reverseString(s1,start,end);
//               start=i+1;
//
//               sb.append(chars);
//               sb.append(' ');
//           }
//       }
//       return  sb.toString();
//
//
//    }

    /**
     * 反转字符串中的单词
     *
     * @param s
     * @return
     */
    public static String reverseWords(String s) {
        String[] strArray = s.split(" ");
        StringBuilder buffer = new StringBuilder("");
        for (String str : strArray) {
            StringBuilder grantBuffer = new StringBuilder(str);
            buffer.append(" ").append(grantBuffer.reverse().toString());
        }
        return buffer.toString().trim();
    }


    public static char[] reverseString(String s, int start, int end) {
        int i = 0, j = end - start - 1;
        char[] chars = s.toCharArray();
        while (i <= j) {
            char tmp = chars[i];
            chars[i] = chars[j];
            chars[j] = tmp;
            i++;
            j--;
        }

        return chars;
     

    }





}
