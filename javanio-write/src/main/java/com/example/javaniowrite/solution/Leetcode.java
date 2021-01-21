package com.example.javaniowrite.solution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author zhaolc
 * @version 1.0
 * @description TODO
 * @createTime 2021年01月18日 10:07:00
 */
public class Leetcode {

    private HashMap<Node, Node> vistedMap = new HashMap<>();

    class Node {
        public int val;
        public Node next;
        public Node random;

        public Node() {
        }

        public Node(int _val, Node _next, Node _random) {
            val = _val;
            next = _next;
            random = _random;
        }
    }


    /**
     * 回溯法，复制带随机指针的节点
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        if (vistedMap.containsKey(head)) {
            return vistedMap.get(head);
        }
        Node node = new Node(head.val, null, null);
        vistedMap.put(head, node);
        node.next = copyRandomList(head.next);
        node.random = copyRandomList(head.random);
        return node;

    }

    /**
     * 二分法搜索旋转排序数组
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        int length = nums.length;
        if (length == 0) {
            return -1;
        }
        if (length == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int left = 0, right = length - 1;
        while (left <= right) {
            int mid = (right + left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                // 目标值在中间值左边
                if (nums[0] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // 目标值在中间值右边
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 螺旋矩阵，按层模拟
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> integerList = new ArrayList<>();
        if (matrix == null || matrix[0].length == 0) {
            return integerList;
        }
        int rows = matrix.length, columns = matrix[0].length;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            // 从左到右
            for (int column = left; column <= right; column++) {
                integerList.add(matrix[top][column]);
            }
            // 从上到下，往下递进一层
            for (int row = top+1; row <= bottom; row++) {
                integerList.add(matrix[row][right]);
            }
            if (left < right && top < bottom) {
                // 从右到左遍历下侧元素,注意往内递进一层
                for (int column = right-1; column > left; column--) {
                    integerList.add(matrix[bottom][column]);
                }
                // 从下到上遍历左侧元素
                for (int row = bottom; row > top; row--) {
                    integerList.add(matrix[row][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return integerList;
    }
}
