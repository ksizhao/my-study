package com.example.javaniowrite.solution;


import javafx.util.Pair;
import org.springframework.util.StringUtils;

import java.util.*;
import java.util.concurrent.CountDownLatch;

/**
 * @author zhaoliancan
 * @version 1.0.0
 * @ClassName Solution.java
 * @Description TODO
 * @createTime 2019年08月23日 08:57:00
 */
public class Solution {


    private TreeNode root;

    private static CountDownLatch latch = new CountDownLatch(10);

    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }

        TreeNode() {
        }
    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    ;


    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public boolean isSymmetric(TreeNode root) {

        return isMirror(root, root);

    }


    /**
     * 向二叉树插入数据
     *
     * @param value
     */
    public void insert(int value) {
        TreeNode newNode = new TreeNode();
        newNode.val = value;
        if (root == null) {
            root = newNode;
        } else {
            // 作为预指针,二叉树构造过程需要指针移动
            TreeNode current = root;
            TreeNode parent;
            while (true) {
                parent = current;
                if (value < current.val) {
                    current = current.left;
                    if (current == null) {
                        parent.left = newNode;
                        return;
                    }
                } else {
                    // 插入右节点
                    current = current.right;
                    if (current == null) {
                        parent.right = newNode;
                        return;
                    }
                }
            }

        }
    }

    /**
     * 把二叉搜索树转换成累加树
     *
     * @param root
     * @return
     */
    public TreeNode convertBST(TreeNode root) {
        if (root != null) {
            dfs(root, 0);
        }
        return root;
    }

    // 逆中序遍历节点计算累计值
    private int dfs(TreeNode node, int sum) {

        if (node == null) {
            return sum;
        }
        // 右子树累计值
        sum = dfs(node.right, sum);
        // 当前节点值+右子树累计值
        node.val = node.val + sum;
        sum = node.val;
        // 当前节点值+左子树累计值
        sum = dfs(node.left, sum);
        return sum;
    }

    /**
     * 如果同时满足下面的条件，两个树互为镜像：
     * <p>
     * 它们的两个根结点具有相同的值。
     * 每个树的右子树都与另一个树的左子树镜像对称。
     *
     * @param t1
     * @param t2
     * @return
     */
    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return t1.val == t2.val && isMirror(t1.left, t2.right) && isMirror(t1.right, t2.left);
    }


    /**
     * 反转单词顺序
     *
     * @param str ""hello world"
     * @return "world hello"
     */
    public static String reverseStr(String str) {
        if (str == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        String[] strings = str.split(" ");
        for (int i = strings.length - 1; i >= 0; i--) {
            sb.append(strings[i]).append(" ");
        }
        return sb.toString();

    }

    /**
     * 两数相加
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // 设置预先指针，使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。
        ListNode pre = new ListNode(0);
        // 变量追踪进位
        int carry = 0;
        ListNode curr = pre;
        while (l1 != null || l2 != null) {
            int x = (l1 == null) ? 0 : l1.val;
            int y = (l2 == null) ? 0 : l2.val;
            int sum = x + y + carry;
            // 更新进位值
            carry = sum / 10;
            // 创建一个数值为 (sum \bmod 10)(summod10) 的新结点
            curr.next = new ListNode(sum % 10);
            // 将当前结点前进到下一个结点。
            curr = curr.next;
            // p,q前进一个节点
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            curr.next = new ListNode(carry);
        }
        return pre.next;

    }


    public ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode curr = pre;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int x = (l1 != null) ? l1.val : 0;
            int y = (l2 != null) ? l2.val : 0;
            int sum = x + y + carry;
            // 更新进位值
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            curr.next = new ListNode(carry);
        }
        return pre.next;

    }

    private static List<String> output = new ArrayList<String>();

    /**
     * 电话号码的字母组合,使用回溯思想
     *
     * @param digits
     * @return
     */
    public static List<String> letterCombinations(String digits) {
        if (digits.length() != 0) {
            backtrack("", digits);

        }
        return output;

    }


    public static void backtrack(String combination, String next_digits) {

        Map<String, String> dataMap = getMap();
        if (next_digits.length() == 0) {
            output.add(combination);
        } else {
            String digiti = next_digits.substring(0, 1);
            String letters = dataMap.get(digiti);
            for (int i = 0; i < letters.length(); i++) {
                String letter = dataMap.get(digiti).substring(i, i + 1);
                backtrack(combination + letter, next_digits.substring(1));
            }
        }
    }

    private static Map<String, String> getMap() {
        Map<String, String> dataMap = new HashMap<>();
        dataMap.put("2", "abc");
        dataMap.put("3", "def");
        dataMap.put("4", "ghi");
        dataMap.put("5", "jkl");
        dataMap.put("6", "mno");
        dataMap.put("7", "pqrs");
        dataMap.put("8", "tuv");
        dataMap.put("9", "wxyz");
        return dataMap;
    }

    /**
     * 反转二叉树,迭代
     *
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {

        if (root == null) {
            return null;
        }
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        nodeQueue.add(root);
        while (!nodeQueue.isEmpty()) {
            TreeNode current = nodeQueue.poll();
            TreeNode tmp = current.left;
            current.left = current.right;
            current.right = tmp;
            if (current.left != null) {
                nodeQueue.add(current.left);
            }
            if (current.right != null) {
                nodeQueue.add(current.right);
            }
        }
        return root;

    }

    /**
     * 递归反转二叉树
     *
     * @param root
     * @return
     */
    public TreeNode invertTree1(TreeNode root) {
        if (root == null) {
            return null;
        } else {

            TreeNode right = invertTree1(root.right);
            TreeNode left = invertTree1(root.left);
            root.left = right;
            root.right = left;
            return root;
        }

    }

    /**
     * 移动零,快慢指针
     *
     * @param nums
     */
    public static void moveZeroes(int[] nums) {
        int curr = 0;
        // 指针指向当前元素，如果新找到的元素不是 0，我们就在最后找到的非 0 元素之后记录它。
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[curr] = nums[i];
                curr++;
            }
        }
        //在 “cur” 索引到达数组的末尾之后，我们现在知道所有非 0 元素都已按原始顺序移动到数组的开头
        for (int i = curr; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    private static Date getYesterday() {
        Date date = new Date();
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        calendar.add(Calendar.DATE, -1);
        date = calendar.getTime();
        return date;
    }

    private Integer max = 0;

    private int depth1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int letfDepth = depth1(root.left);
        int rightDepth = depth1(root.right);
        max = Math.max(letfDepth + rightDepth, max);
        return Math.max(letfDepth, rightDepth) + 1;
    }

    /**
     * 递归求二叉树的直径，一个节点的最大直径 = 它左树的高度 +  它右树的高度，二叉树直径=节点最大直径+根节点
     *
     * @param root
     * @return
     */
    public int diameterOfBinaryTree(TreeNode root) {

        depth(root);
        return max;
    }

    private int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = depth(root.left);
        int rightDepth = depth(root.right);
        max = Math.max(leftDepth + rightDepth, max);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    /**
     * 合并二叉树，使用递归
     *
     * @param t1
     * @param t2
     * @return
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {

        if (t1 == null) {
            return t2;
        }
        if (t2 == null) {
            return t1;
        }
        TreeNode treeNode = new TreeNode(t1.val + t2.val);
        treeNode.left = mergeTrees(t1.left, t2.left);
        treeNode.right = mergeTrees(t1.right, t2.right);
        return treeNode;
    }

    public static Thread getThread() {
        return new Thread(() -> {
            try {
                System.out.println("线程开始等待");
                latch.await();
                System.out.println("线程等待结束");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 验证回文串
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            // 跳过非法字符
            while (i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                i++;

            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                j--;
            }
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) {
                return false;
            }
            i++;
            j--;
        }
        return true;

    }

    public int lengthOfLastWord(String s) {
        if (StringUtils.isEmpty(s)) {
            return 0;
        }
        String[] strings = s.split(" ");
        if (strings.length == 0) {
            return 0;
        } else {
            String s1 = strings[strings.length - 1];
            return s1.length();
        }

    }

    public static int lengthOfLastWord1(String s) {
        String s1 = s.trim();
        int end = s1.length() - 1;
        if (end < 0) {
            return 0;
        }
        int start = end;
        while (start >= 0 && s.charAt(start) != ' ') {
            start--;
        }
        return end - start;
    }

    /**
     * 二分法查找元素插入位置
     *
     * @param nums
     * @param target
     * @return
     */
    public static int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    /**
     * 杨辉三角
     *
     * @param numRows
     * @return
     */
    public static List<List<Integer>> generate(int numRows) {

        List<List<Integer>> lists = new ArrayList<>();
        if (numRows == 0) {
            return lists;
        }
        lists.add(new ArrayList<>());
        lists.get(0).add(1);
        for (int numRow = 1; numRow < numRows; numRow++) {
            List<Integer> row = new ArrayList<>();
            row.add(1);
            List<Integer> preRow = lists.get(numRow - 1);
            for (int j = 1; j < numRow; j++) {
                row.add(preRow.get(j - 1) + preRow.get(j));
            }
            row.add(1);
            lists.add(row);
        }
        return lists;

    }

    public static List<Integer> getRow(int rowIndex) {

        List<Integer> pre = new ArrayList<>();
        pre.add(1);
        for (int i = 0; i < rowIndex; i++) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);

            for (int j = 0; j < i; j++) {
                tmp.add(pre.get(j) + pre.get(j + 1));
            }
            tmp.add(1);
            pre = tmp;
        }
        return pre;

    }

    public void rotate1(int[] nums, int k) {
        int pre, tmp;
        for (int i = 0; i < k; i++) {
            pre = nums[nums.length - 1];
            for (int j = 0; j < nums.length; j++) {
                tmp = pre;
                pre = nums[j];
                nums[j] = tmp;
            }
        }
    }


    /**
     * 旋转数组,旋转k次，则k%n个元素被移到头部，剩余的往后移动
     *
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {

        if (nums.length > 0) {
            k %= nums.length;
            reverse(nums, 0, nums.length - 1);
            reverse(nums, 0, k - 1);
            reverse(nums, k, nums.length - 1);
        }
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
            start++;
            end--;
        }
    }


    /**
     * 数组是有序数组，使用快慢指针寻找两数之和
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {

        int low = 0, high = numbers.length - 1;
        while (low < high) {
            int sum = numbers[low] + numbers[high];
            if (sum < target) {
                low++;
            } else if (sum == target) {
                return new int[]{low + 1, high + 1};
            } else {
                high--;
            }
        }
        throw new RuntimeException("error");
    }

    /**
     * 缺失的数字
     *
     * @param nums
     * @return
     */
    public static int missingNumber(int[] nums) {

        Arrays.sort(nums);
        int target = 0;
        for (int i = 0; i < nums.length; i++) {
            if (target == nums[i]) {
                target++;
            } else {
                break;
            }
        }
        return target;
        /**方法二 高斯求和公式**/
//        int exceptSum = nums.length * (nums.length + 1) / 2;
//        int sum = 0;
//        for (int i = 0; i < nums.length; i++) {
//            sum += nums[i];
//        }
//        return exceptSum - sum;


    }

    public static int missingNumber1(int[] nums) {
        int exceptSum = nums.length * (nums.length + 1) / 2;
        int realSum = 0;
        for (int i = 0; i < nums.length; i++) {
            realSum += nums[i];
        }
        return exceptSum - realSum;
    }

    /**
     * 维护一个哈希表，里面始终最多包含 k 个元素，当出现重复值时则说明在 k 距离内存在重复元素
     * 每次遍历一个元素则将其加入哈希表中，如果哈希表的大小大于 k，则移除最前面的数字
     *
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> integerSet = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (integerSet.contains(nums[i])) {
                return true;
            }
            integerSet.add(nums[i]);
            if (integerSet.size() > k) {
                integerSet.remove(nums[i - k]);
            }
        }
        return false;
    }

    /**
     * 遍历
     * count 为当前元素峰值，maxCount为最大峰值
     * 初始化 count = 1
     * 从 0 位置开始遍历，遍历时根据前后元素状态判断是否递增，递增则 count++，递减则 count=1
     * 如果 count>maxCount，则更新 maxCount
     * 直到循环结束
     *
     * @param nums
     * @return
     */
    public static int findLengthOfLCIS(int[] nums) {

        if (nums.length <= 1) {
            return nums.length;
        }
        int count = 1;
        int maxCount = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i + 1] > nums[i]) {
                count++;
            } else {
                count = 1;
            }
            maxCount = count > maxCount ? count : maxCount;
        }
        return maxCount;

    }

    /**
     * 最大连续1的个数
     *
     * @param nums
     * @return
     */
    public static int findMaxConsecutiveOnes(int[] nums) {
        int count = 0;
        int maxCount = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                count++;
                maxCount = Math.max(count, maxCount);
            } else {
                count = 0;
            }
        }
        return maxCount;

    }

    /**
     * 按奇偶排序数组
     *
     * @param A
     * @return
     */
    public static int[] sortArrayByParity(int[] A) {

        int[] arrays = new int[A.length];
        int i = 0, k = A.length - 1;
        for (int j = 0; j < A.length; j++) {
            if (A[j] % 2 == 0) {
                arrays[i] = A[j];
                i++;
            } else {
                arrays[k] = A[j];
                k--;
            }
        }
        return arrays;
    }

    /**
     * 盛最多水的容器
     *
     * @param height
     * @return
     */
    public static int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;

    }

    /**
     * 旋转图像
     * 先转置矩阵，然后翻转每一行
     *
     * @param matrix
     */
    public static void rotate(int[][] matrix) {

        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = tmp;
            }
        }
    }

    /**
     * 寻找中心索引
     *
     * @param nums
     * @return
     */
    public int pivotIndex(int[] nums) {

        int totalSum = 0;
        int leftSum = 0;
        for (int i = 0; i < nums.length; i++) {
            totalSum += nums[i];
        }
        for (int i = 0; i < nums.length; i++) {
            if (leftSum * 2 == totalSum - nums[i]) {
                return i;
            }
            leftSum += nums[i];
        }
        return -1;
    }

    /**
     * 逐个枚举，空集的幂集只有空集，每增加一个元素，让之前幂集中的每个集合，追加这个元素，就是新增的子集。
     *
     * @param nums
     * @return
     */
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>());
        for (Integer n : nums) {
            int size = res.size();
            for (int i = 0; i < size; i++) {
                List<Integer> newSub = new ArrayList<>(res.get(i));
                newSub.add(n);
                res.add(newSub);
            }
        }
        return res;

    }

    /**
     * 寻找多次出现的数字
     *
     * @param nums
     * @return
     */
    public static List<Integer> findDuplicates(int[] nums) {
        // 方法一
//        Set<Integer> integerSet = new HashSet<>();
//        List<Integer> integerList = new LinkedList<>();
//        for (int i = 0; i < nums.length; i++) {
//            if (integerSet.contains(nums[i])) {
//                integerList.add(nums[i]);
//            } else {
//                integerSet.add(nums[i]);
//            }
//        }
//        return integerList;
        List<Integer> arrayList = new ArrayList<>();
        int len = nums.length;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] <= len && (nums[i] - 1) != nums[i]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < len; i++) {
            if (nums[i] - 1 != i) {
                arrayList.add(nums[i]);
            }
        }
        return arrayList;
    }

    private static void swap(int[] nums, int index1, int index2) {
        if (index1 == index2) {
            return;
        }
        nums[index1] = nums[index1] ^ nums[index2];
        nums[index2] = nums[index1] ^ nums[index2];
        nums[index1] = nums[index1] ^ nums[index2];
    }


    private int[] preorder;
    private int[] inorder;
    private int[] postorder;
    private int preIndex = 0;
    private int lastIndex = 0;
    private HashMap<Integer, Integer> inorderMap = new HashMap<>();
    private HashMap<Integer, Integer> postorderMap = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {

        if (preorder == null || inorder == null) {
            return null;
        }
        this.preorder = preorder;
        this.inorder = inorder;
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        return buildTree(0, inorder.length - 1);

    }

    private TreeNode buildTree(int start, int end) {
        if (start > end) {
            return null;
        }
        int preVal = preorder[preIndex];
        TreeNode root = new TreeNode(preVal);
        int index = inorderMap.get(preVal);
        preIndex++;
        root.left = buildTree(start, index - 1);
        root.right = buildTree(index + 1, end);
        return root;
    }

    public TreeNode buildTree1(int[] inorder, int[] postorder) {
        if (postorder == null || inorder == null) {
            return null;
        }
        this.inorder = inorder;
        this.postorder = postorder;
        lastIndex = postorder.length - 1;
        for (int i = 0; i < postorder.length; i++) {
            postorderMap.put(postorder[i], i);
        }
        return buildTree1(0, inorder.length - 1, 0, postorder.length - 1);
    }

    private TreeNode buildTree1(int inLeft, int inRight, int postLeft, int postRight) {
        if (inLeft > inRight || postLeft > postRight) {
            return null;
        }
        int rootVal = postorder[lastIndex];
        TreeNode root = new TreeNode(rootVal);
        int index = postorderMap.get(rootVal);
        preIndex--;
        root.left = buildTree1(inLeft, index - 1, postLeft, postRight - inRight + index - 1);
        root.right = buildTree1(index + 1, inRight, postRight - inRight + index, postRight - 1);
        return root;
    }


    List<List<Integer>> res = new ArrayList<>();
    int[] candidates;
    int len;

    /**
     * 回溯+剪枝算法计算组合总和
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        if (candidates.length == 0) {
            return res;
        }
        this.candidates = candidates;
        this.len = candidates.length;
        Arrays.sort(candidates);
        findResult(target, 0, new Stack<>());
        return res;
    }

    public void findResult(int residue, int start, Stack<Integer> pre) {

        if (residue == 0) {
            // Java 中可变对象是引用传递，因此需要将当前 path 里的值拷贝出来
            res.add(new ArrayList<>(pre));
            return;
        }
        // residue - candidates[i] 表示下一轮的剩余，如果下一轮的剩余都小于 0 ，就没有必要进行后面的循环了
        // 这一点基于原始数组是排序数组的前提，因为如果计算后面的剩余，只会越来越小
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            pre.add(candidates[i]);
            // 【关键】因为元素可以重复使用，这里递归传递下去的是 i 而不是 i + 1
            findResult(residue - candidates[i], i, pre);
            pre.pop();
        }

    }

    /**
     * 动态规划计算组合总和
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum1(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, Set<List<Integer>>> map = new HashMap<>();
        //对candidates数组进行排序
        Arrays.sort(candidates);
        int len = candidates.length;
        for (int i = 1; i <= target; i++) {
            //初始化map
            map.put(i, new HashSet<>());
            //对candidates数组进行循环
            for (int j = 0; j < len && candidates[j] <= target; j++) {
                if (i == candidates[j]) {
                    //相等即为相减为0的情况，直接加入set集合即可
                    List<Integer> temp = new ArrayList<>();
                    temp.add(i);
                    map.get(i).add(temp);
                } else if (i > candidates[j]) {
                    //i-candidates[j]是map的key
                    int key = i - candidates[j];
                    //使用迭代器对对应key的set集合进行遍历
                    //如果candidates数组不包含这个key值，对应的set集合会为空，故这里不需要做单独判断
                    for (Iterator iterator = map.get(key).iterator(); iterator.hasNext(); ) {
                        List list = (List) iterator.next();
                        //set集合里面的每一个list都要加入candidates[j]，然后放入到以i为key的集合中
                        List tempList = new ArrayList<>();
                        tempList.addAll(list);
                        tempList.add(candidates[j]);
                        //排序是为了通过set集合去重
                        Collections.sort(tempList);
                        map.get(i).add(tempList);
                    }
                }
            }
        }
        result.addAll(map.get(target));
        return result;
    }

    /**
     * 动态规划计算不同路径
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        //法一
//
//        int[][] dp = new int[m][n];
//        for (int i = 0; i < m; i++) {
//            dp[i][0] = 1;
//        }
//        for (int i = 0; i < n; i++) {
//            dp[0][i] = 1;
//        }
//
//        for (int i = 1; i < m; i++) {
//            for (int j = 1; j < n; j++) {
//                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
//            }
//        }
//        return dp[m - 1][n - 1];
        // 法二
        int[] cur = new int[n];
        Arrays.fill(cur, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] += cur[j - 1];
            }
        }
        return cur[n - 1];


    }

    /**
     * 动态规划计算最小路径和
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {

        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = grid[i][j] + dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] = grid[i][j] + dp[i - 1][j];
                } else {
                    dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m - 1][n - 1];
        // 倒序
//        for (int i = m - 1; i >= 0; i--) {
//            for (int j = n - 1; j >= 0; j--) {
//                if (i == grid.length - 1 && j != grid[0].length - 1) {
//                    dp[i][j] = grid[i][j] + dp[i][j + 1];
//                } else if (i != grid.length - 1 && j == grid[0].length - 1) {
//                    dp[i][j] = grid[i][j] + dp[i + 1][j];
//                } else if (i != grid.length - 1 && j != grid[0].length - 1) {
//                    dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
//                } else {
//                    dp[i][j] = grid[i][j];
//                }
//            }
//        }
//        return dp[0][0];
    }

    /**
     * 除自身以外数组的乘积
     *
     * @param nums
     * @return
     */
    public static int[] productExceptSelf(int[] nums) {

        int p = 1, q = 1;
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            res[i] = p;
            p *= nums[i];
        }
        for (int i = nums.length - 1; i > 0; i--) {
            q *= nums[i];
            res[i - 1] *= q;
        }
        return res;
    }

    public int[] searchRange(int[] nums, int target) {

        int p = -1, q = -1;
        for (int i = 0; i < nums.length; i++) {
            if (target == nums[i]) {
                p = i;
            }
        }
        for (int j = nums.length - 1; j >= 0; j--) {
            if (target == nums[j]) {
                q = j;
            }
        }
        return new int[]{q, p};

    }

    private static int extremeInsertionIndex(int[] nums, int target, boolean left) {
        int low = 0;
        int high = nums.length - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (nums[mid] > target || (left && nums[mid] == target)) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    /**
     * 二分法查找元素位置
     *
     * @param nums
     * @param target
     * @return
     */
    public static int[] searchRange1(int[] nums, int target) {
        int[] targetRange = {-1, -1};
        int leftIndex = extremeInsertionIndex(nums, target, true);
        if (leftIndex == nums.length || nums[leftIndex] != target) {
            return targetRange;
        }
        targetRange[0] = leftIndex;
        targetRange[1] = extremeInsertionIndex(nums, target, false) - 1;
        return targetRange;

    }

    /**
     * 动态规划构造搜索树
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    /**
     * 颜色分类，荷兰国旗问题
     *
     * @param nums
     */
    public static void sortColors(int[] nums) {

        int p0 = 0, p2 = nums.length - 1, curr = 0;
        int temp;
        while (curr <= p2) {
            if (nums[curr] == 0) {
                temp = nums[curr];
                nums[curr] = nums[p0];
                nums[p0] = temp;
                p0++;
                curr++;
            } else if (nums[curr] == 2) {
                temp = nums[curr];
                nums[curr] = nums[p2];
                nums[p2] = temp;
                p2--;
            } else {
                curr++;
            }
        }

    }

    /**
     * 三角形最小路径和
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        int row = triangle.size();
        int[] minLen = new int[row + 1];
        for (int i = row - 1; i >= 0; i--) {
            // 第i行有i+1个数字
            for (int j = 0; j <= i; j++) {
                minLen[j] = Math.min(minLen[j], minLen[j + 1]) + triangle.get(i).get(j);
            }
        }
        return minLen[0];

    }

    /**
     * 字母异位词分组
     *
     * @param strs
     * @return
     */
    public static List<List<String>> groupAnagrams(String[] strs) {

        if (strs.length == 0) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> hashMap = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            char[] chars = strs[i].toCharArray();
            Arrays.sort(chars);
            String tmpStr = String.valueOf(chars);
            if (!hashMap.containsKey(tmpStr)) {
                hashMap.put(tmpStr, new ArrayList<>());
                hashMap.get(tmpStr).add(strs[i]);
            } else {
                hashMap.get(tmpStr).add(strs[i]);
            }
        }
        return new LinkedList<>(hashMap.values());
    }

    /**
     * 计数质数
     *
     * @param n
     * @return
     */
    public static int countPrimes(int n) {

        int count = 0;
        boolean[] signs = new boolean[n];
        for (int i = 2; i < n; i++) {
            if (!signs[i]) {
                count++;
                for (int j = i + i; j < n; j += i) {
                    //排除不是质数的数
                    signs[j] = true;
                }
            }
        }
        return count;

    }

    /**
     * 迭代求链表反转
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {

        ListNode pre = null;
        ListNode curr = head;
        ListNode next = null;
        while (curr != null) {
            next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return head;

    }

    /**
     * 回文链表
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        List<Integer> integerList = new LinkedList<>();
        ListNode curr = head;
        while (curr != null) {
            integerList.add(curr.val);
            curr = curr.next;
        }
        int first = 0, tail = integerList.size() - 1;
        while (first < tail) {
            if (!integerList.get(first).equals(integerList.get(tail))) {
                return false;
            } else {
                first++;
                tail--;
            }
        }
        return true;
    }

    /**
     * 数组中数字出现的次数
     *
     * @param nums
     * @return
     */
    public int[] singleNumbers(int[] nums) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            k ^= nums[i];
        }

        int mask = 1;
        while ((k & mask) == 0) {
            mask <<= 1;
        }
        int a = 0, b = 0;
        for (int i = 0; i < nums.length; i++) {
            if ((nums[i] & mask) == 0) {
                a ^= nums[i];
            } else {
                b ^= nums[i];
            }
        }
        return new int[]{a, b};
    }

    /**
     * 调整数组顺序使奇数位于偶数前面
     *
     * @param nums
     * @return
     */
    public static int[] exchange(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            while (left < right && (nums[left] & 1) != 0) {
                left++;
            }
            while (left < right && (nums[right] & 1) == 0) {
                right--;
            }
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
        }
        return nums;
    }


    private static void swap(int a, int b) {
        int temp = b;
        b = a;
        a = temp;
    }

    public int firstUniqChar(String s) {

        HashMap<Character, Integer> hashMap = new HashMap<>();
        char[] chars = s.toCharArray();
        for (char c : chars) {
            hashMap.put(c, hashMap.getOrDefault(c, 0) + 1);
        }
        for (int i = 0; i < chars.length; i++) {
            if (hashMap.get(chars[i]) == 1) {
                return i;
            }
        }
        return -1;

    }

    /**
     * 验证二叉搜索树
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack();
        double inorder = -Double.MAX_VALUE;

        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            // If next element in inorder traversal
            // is smaller than the previous one
            // that's not BST.
            if (root.val <= inorder) {
                return false;
            }
            inorder = root.val;
            root = root.right;
        }
        return true;
    }


    /**
     * 判断是否为欢乐数
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        HashSet<Integer> hashSet = new HashSet<>();
        while (!hashSet.contains(n) && n > 0) {
            hashSet.add(n);
            n = getNext(n);
        }
        return n == 1;

    }

    /**
     * 获取位数之和
     *
     * @param n
     * @return
     */
    private int getNext(int n) {
        int totalNum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalNum += d * d;
        }
        return totalNum;
    }

    public int findDuplicate(int[] nums) {

        HashSet<Integer> hashSet = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (hashSet.contains(nums[i])) {
                return nums[i];
            }
            hashSet.add(nums[i]);
        }
        return -1;
    }

    /**
     * x的平方根，用二分法实现
     *
     * @param x
     * @return
     */
    public static int mySqrt(int x) {

        if (x == 0) {
            return 0;
        }
        long left = 1;
        long right = x / 2;
        while (left < right) {
            long mid = left + (right - left + 1) / 2;
            long qure = mid * mid;
            if (qure > x) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return (int) left;

    }

    /**
     * 只出现一次的数字
     *
     * @param nums
     * @return
     */
    public static int singleNumber(int[] nums) {
        if (nums == null) {
            return 0;
        }
        int size = nums.length;
        int result = 0;
        for (int i = 0; i < size; i++) {

            result = nums[i] ^ result;
        }
        return result;

    }

    /**
     * 返回字符串中第一个唯一的字符
     *
     * @param s
     * @return
     */
    public char firstUniqChar1(String s) {
        if (s.isEmpty()) {
            return ' ';
        }
        char[] chars = s.toCharArray();
        Map<Character, Integer> hashMap = new LinkedHashMap<>();
        for (int i = 0; i < chars.length; i++) {
            hashMap.put(chars[i], hashMap.getOrDefault(chars[i], 0) + 1);
        }
        for (int i = 0; i < chars.length; i++) {
            if (hashMap.get(chars[i]) == 1) {
                return chars[i];
            }
        }
        return ' ';
    }

    /**
     * 单词拆分 ，动态规划
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> stringSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && stringSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * 前 K 个高频元素
     *
     * @param nums
     * @param k
     * @return
     */
//    public static List<Integer> topKFrequent(int[] nums, int k) {
//
//        HashMap<Integer, Integer> hashMap = new HashMap<>();
//        for (int i = 0; i < nums.length; i++) {
//            hashMap.put(nums[i], hashMap.getOrDefault(nums[i], 0) + 1);
//        }
//
//        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(new Comparator<Integer>() {
//            @Override
//            public int compare(Integer o1, Integer o2) {
//                return hashMap.get(o1) - hashMap.get(o2);
//            }
//        });
//
//        for (Map.Entry<Integer, Integer> entry : hashMap.entrySet()) {
//            if (priorityQueue.size() < k) {
//                priorityQueue.add(entry.getKey());
//            } else if (entry.getValue() > hashMap.get(priorityQueue.peek())) {
//                priorityQueue.remove();
//                priorityQueue.add(entry.getKey());
//            }
//        }
//
//        List<Integer> integerList = new ArrayList<>();
//        while (!priorityQueue.isEmpty()) {
//            integerList.add(priorityQueue.poll());
//        }
//        return integerList;
//    }

    /**
     * 全排列
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        List<Integer> output = new LinkedList<>();
        for (int num : nums) {
            output.add(num);
        }
        backTrack(nums.length, res, output, 0);
        return res;
    }


    private void backTrack(int n, List<List<Integer>> res, List<Integer> output, int first) {

        if (first == n) {
            res.add(new LinkedList<>(output));
        }
        for (int i = first; i < n; i++) {
            Collections.swap(output, i, first);
            backTrack(n, res, output, first + 1);
            Collections.swap(output, i, first);
        }

    }

    /**
     * 反转二进制数
     *
     * @param n
     * @return
     */
    public static int reverseBits(int n) {

        int res = 0, count = 0;
        while (count < 32) {
            res = res << 1;
            res |= (n & 1);
            n = n >> 1;
            count++;
        }
        return res;

    }

    /**
     * 位操作求和
     *
     * @param n
     * @return
     */
    public int sumNums(int n) {
        boolean result = n > 1 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    /**
     * 使用指定字符替换空格
     *
     * @param s
     * @return
     */
    public static String replaceSpace(String s) {
        char[] chars = s.toCharArray();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] == ' ') {
                stringBuilder.append("%20");
            } else {
                stringBuilder.append(chars[i]);
            }

        }
        return stringBuilder.toString();

    }

    public int findDuplicate1(int[] nums) {

        List<Integer> integerList = new ArrayList<>();
        for (int num : nums) {
            integerList.add(num);
        }
        Collections.sort(integerList);
        for (int i = 0; i < integerList.size(); i++) {
            if (integerList.get(i).equals(integerList.get(i - 1))) {
                return integerList.get(i);
            }
        }
        return 0;
    }

    /**
     * 3的幂次方
     *
     * @param n
     * @return
     */
    public boolean isPowerOfThree(int n) {

        if (n < 1) {
            return false;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        return n == 1;
    }

    /**
     * 排序链表
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {

        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        // 切断链表
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left != null && right != null) {

            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        h.next = left != null ? left : right;
        return res.next;


    }

    public int hammingWeight(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;

    }

    /**
     * 动态规划，求解完全平方数
     *
     * @param n
     * @return
     */
    public int numSquares(int n) {
        // 默认初始化值都为0
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            // 最坏的情况就是每次+1
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j++) {
                // 动态转移方程
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    /**
     * 回溯法分割回文串
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        int length = s.length();
        for (int len = 1; len <= length; len++) {
            for (int i = 0; i <= s.length() - len; i++) {
                dp[i][i + len - 1] = s.charAt(i) == s.charAt(i + len - 1) && (len < 3 || dp[i + 1][i + len - 2]);
            }
        }
        List<List<String>> ans = new ArrayList<>();
        partitionHelper(s, 0, dp, new ArrayList<>(), ans);
        return ans;
    }

    private void partitionHelper(String s, int start, boolean[][] dp, List<String> temp, List<List<String>> res) {
        //到了空串就加到最终的结果中
        if (start == s.length()) {
            res.add(new ArrayList<>(temp));
        }
        //在不同位置切割
        for (int i = start; i < s.length(); i++) {
            //如果是回文串就加到结果中
            if (dp[start][i]) {
                String left = s.substring(start, i + 1);
                temp.add(left);
                partitionHelper(s, i + 1, dp, temp, res);
                temp.remove(temp.size() - 1);
            }

        }
    }


    /**
     * 数组中的第K个最大元素
     *
     * @param nums
     * @param k
     * @return
     */
    public static int findKthLargest(int[] nums, int k) {

        Arrays.sort(nums);
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        int j = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            hashMap.put(j, nums[i]);
            j++;
        }
        return hashMap.get(k);
    }


    /**
     * 有效的字母异位词
     *
     * @param s
     * @param t
     * @return
     */
    public static boolean isAnagram(String s, String t) {

        if (s.length() != t.length()) {
            return false;
        }
        int[] counter = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counter[s.charAt(i) - 'a']++;
            counter[t.charAt(i) - 'a']--;
        }
        for (int count : counter) {
            if (count != 0) {
                return false;
            }
        }
        return true;

    }

    /**
     * 第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
     * 第二步：计算进位值，得到1010，相当于各位进行与操作得到101，再向左移一位得到1010，(101&111)<<1。
     * 第三步重复上述两步，各位相加 010^1010=1000，进位值为100=(010 & 1010)<<1。
     * 继续重复上述两步：1000^100 = 1100，进位值为0，跳出循环，1100为最终结果。
     * 结束条件：进位为0，即a为最终的求和结果。
     *
     * @param a
     * @param b
     * @return
     */
    public int getSum(int a, int b) {
        while (b != 0) {
            int temp = a ^ b;
            b = (a & b) << 1;
            a = temp;
        }
        return a;

    }

    /**
     * 摩尔投票，求众数
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {

        int votes = 0, x = 0;
        for (int num : nums) {
            if (votes == 0) {
                x = num;
            }
            votes += x == num ? 1 : -1;
        }
        return x;
    }

    /**
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * <p>
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额
     * <p>
     * 来源：力扣（LeetCode）
     * 链接：https://leetcode-cn.com/problems/house-robber
     * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     * 动态规划方程：dp[n] = MAX( dp[n-1], dp[n-2] + num )求打家劫舍
     *
     * @param nums
     * @return
     */
    public static int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 2; i <= nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[nums.length];
//        int length = nums.length;
//        if (length == 0) {
//            return 0;
//        }
//        if(length==1){
//            return nums[0];
//        }
//        int[] dp = new int[length];
//        dp[0] = nums[0];
//        dp[1] = Math.max(nums[0], nums[1]);
//        for (int i = 2; i < length; i++) {
//            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
//        }
//        return dp[length - 1];


    }

    /**
     * 奇偶链表
     *
     * @param head
     * @return
     */
    public ListNode oddEvenList(ListNode head) {

        if (head == null) {
            return null;
        }
        ListNode odd = head, even = head.next, evenHead = even;
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    /**
     * 二叉搜索树中第K小的元素
     *
     * @param root
     * @param k    广度优先搜索（BFS）
     *             在这个策略中，我们逐层，从上到下扫描整个树,BST 的中序遍历是升序序列
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        LinkedList<TreeNode> stack = new LinkedList<TreeNode>();
        while (true) {
            while (root != null) {
                stack.add(root);
                root = root.left;
            }
            root = stack.removeLast();
            if (k-- == 0) {
                return root.val;
            }
            root = root.right;
        }
    }

    private class LargerNumberComparator implements Comparator<String> {

        @Override
        public int compare(String o1, String o2) {
            String str1 = o1 + o2;
            String st2 = o2 + o1;
            return st2.compareTo(str1);
        }
    }

    public String largestNumber(int[] nums) {

        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strings, new LargerNumberComparator());
        if ("0".equals(nums[0])) {
            return "0";
        }
        String largestNumber = "";
        for (String str : strings) {
            largestNumber += str;
        }
        return largestNumber;
    }

    /**
     * 求股票的最大值
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {

        if (prices.length == 0) {
            return 0;
        }
        int maxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        return maxProfit;
//        int minPrice = Integer.MAX_VALUE;
//        for (int i = 0; i < prices.length; i++) {
//            minPrice = Math.min(minPrice, prices[i]);
//            maxProfit = Math.max(maxProfit, prices[i] - minPrice);
//        }
//
//        return maxProfit;
    }

    public static void bubbleSort(int[] array) {
        if (array.length == 0) {
            return;
        }
        int lastIndex = 0;
        int sortBorder = array.length - 1;
        for (int i = 0; i < array.length - 1; i++) {
            boolean sorted = true;
            for (int j = 0; j < sortBorder; j++) {

                if (array[j + 1] < array[j]) {
                    int temp = array[j + 1];
                    array[j + 1] = array[j];
                    array[j] = temp;
                    sorted = false;
                    lastIndex = j;
                }
            }
            sortBorder = lastIndex;
            if (sorted) {
                break;
            }
        }
    }

    /**
     * 快速排序
     *
     * @param array
     * @param startIndex
     * @param endIndex
     */
    public static void quickSort(int[] array, int startIndex, int endIndex) {
        if (startIndex >= endIndex) {
            return;
        }
        int partition = partition(array, startIndex, endIndex);
        quickSort(array, startIndex, partition - 1);
        quickSort(array, partition + 1, endIndex);
    }

    /**
     * 单边循环法
     *
     * @param array
     * @param startIndex
     * @param endIndex
     * @return
     */
    public static int partition(int[] array, int startIndex, int endIndex) {
        int pviot = array[startIndex];
        int mark = startIndex;
        for (int i = startIndex + 1; i < endIndex; i++) {
            if (array[i] < pviot) {
                mark++;
                int temp = array[mark];
                array[mark] = array[i];
                array[i] = temp;
            }
        }
        array[startIndex] = array[mark];
        array[mark] = pviot;
        return mark;
    }

    /**
     * 删除链表的倒数第N个节点
     * 一次 遍历法 第一个指针从列表的开头向前移动 n+1n+1 步，而第二个指针将从列表的开头出发。现在，
     * 这两个指针被 nn 个结点分开。我们通过同时移动两个指针向前来保持这个恒定的间隔，直到第一个指针到达最后一个结点。
     * 此时第二个指针将指向从最后一个结点数起的第 nn 个结点。我们重新链接第二个指针所引用的结点的 next 指针指向该结点的下下个结点。
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {

        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode first = pre;
        ListNode second = pre;
        // Advances first pointer so that the gap between first and second is n nodes apart
        for (int i = 0; i < n + 1; i++) {
            first = first.next;
        }
        // Move first to the end, maintaining the gap
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return pre.next;
    }

    /**
     * 罗马数字转整数
     *
     * @param s
     * @return
     */
    public static int romanToInt(String s) {

        int sum = 0;
        int preNum = getValue(s.charAt(0));
        for (int i = 1; i < s.length(); i++) {
            int num = getValue(s.charAt(i));
            if (preNum < num) {
                sum -= preNum;
            } else {
                sum += preNum;
            }
            preNum = num;
        }
        sum += preNum;
        return sum;

    }

    private static int getValue(char str) {
        switch (str) {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;
        }
    }

    public static int titleToNumber(String s) {
        int sum = 0;
        for (int i = 0; i < s.length(); i++) {
            int num = s.charAt(i) - 'A' + 1;
            sum = sum * 26 + num;
        }
        return sum;

    }

    /**
     * 二叉树层序遍历
     *
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrder(TreeNode root) {

        List<List<Integer>> lists = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            int n = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                TreeNode treeNode = queue.poll();
                list.add(treeNode.val);
                if (treeNode.left != null) {
                    queue.add(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.add(treeNode.right);
                }
            }
            lists.add(list);
        }
        return lists;
    }

    /**
     * 二叉树中序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {

        List<Integer> integerList = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.add(node);
                node = node.left;
            }
            if (!stack.isEmpty()) {
                node = stack.pop();
                integerList.add(node.val);
                node = node.right;
            }

        }
        return integerList;

    }

    public TreeNode sortedArrayToBST(int[] nums) {

        return sortTree(nums, 0, nums.length - 1);
    }

    /**
     * 递归求有序数组转二叉平衡树
     *
     * @param nums
     * @param left
     * @param right
     * @return
     */
    private TreeNode sortTree(int[] nums, int left, int right) {

        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortTree(nums, left, mid - 1);
        root.right = sortTree(nums, mid + 1, right);
        return root;
    }

    public static List<String> fizzBuzz(int n) {

        List<String> stringList = new LinkedList<>();
        for (int i = 0; i <= n; i++) {
            if (i % 3 == 0 && i % 5 == 0) {
                stringList.add("FizzBuzz");
            } else if (i % 3 == 0) {
                stringList.add("Fizz");
            } else if (i % 5 == 0) {
                stringList.add("Buzz");
            } else {
                stringList.add(String.valueOf(i));
            }
        }
        return stringList;

    }


    Map<Integer, TreeNode> parent = new HashMap<Integer, TreeNode>();
    Set<Integer> visited = new HashSet<Integer>();

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        dfs(root);
        while (p != null) {
            visited.add(p.val);
            p = parent.get(p.val);
        }
        while (q != null) {
            if (visited.contains(q.val)) {
                return q;
            }
            visited.add(q.val);
            q = parent.get(q.val);
        }
        return null;
    }

    private void dfs(TreeNode root) {
        if (root.left != null) {
            parent.put(root.left.val, root);
            dfs(root.left);
        }
        if (root.right != null) {
            parent.put(root.right.val, root);
            dfs(root.right);
        }
    }

    /**
     * 有序矩阵中第K小的元素
     *
     * @param matrix
     * @param k
     * @return
     */
    public int kthSmallest(int[][] matrix, int k) {

        int left = matrix[0][0];
        int n = matrix.length;
        int right = matrix[n - 1][n - 1];
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (check(matrix, mid, n, k)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean check(int[][] matrix, int mid, int n, int k) {
        int i = n - 1;
        int j = 0;
        int num = 0;
        while (i >= 0 && j < n) {
            if (matrix[i][j] <= mid) {
                num += i + 1;
                j++;
            } else {
                i--;
            }
        }
        return num >= k;
    }

    /**
     * 最长回文子串
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        // s.charAt(i) 每次都会检查数组下标越界，因此先转换成字符数组
        char[] chars = s.toCharArray();
        // 枚举所有长度大于 1 的子串 charArray[i..j]
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                if (j - i + 1 > maxLen && validPalindromic(chars, i, j)) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);

    }

    /**
     * 判断是否为回文串
     *
     * @param charArray
     * @param left
     * @param right
     * @return
     */
    private static boolean validPalindromic(char[] charArray, int left, int right) {
        while (left < right) {
            if (charArray[left] != charArray[right]) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    /**
     * 乘积最大子数组
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int max = Integer.MIN_VALUE, imax = 1, imin = 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                int temp = imax;
                imax = imin;
                imin = temp;
            }
            imax = Math.max(nums[i] * imax, nums[i]);
            imin = Math.min(nums[i] * imin, nums[i]);
            max = Math.max(max, imax);
        }
        return max;
    }

    /**
     * 环形链表
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {

        Set<ListNode> integerSet = new HashSet<>();
        while (head != null) {
            if (integerSet.contains(head)) {
                return true;
            } else {
                integerSet.add(head);
            }
            head = head.next;
        }
        return false;

    }

    /**
     * 前K个高频元素
     *
     * @param nums
     * @param k
     * @return
     */
    public static int[] topKFrequent(int[] nums, int k) {
        if (nums.length == 0) {
            return null;
        }
        HashMap<Integer, Integer> integerMap = new HashMap<>();
        for (int num : nums) {
            integerMap.put(num, integerMap.getOrDefault(num, 0) + 1);
        }
        // 倒序排列的优先级队列
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<Integer>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return integerMap.get(o1) - integerMap.get(o2);
            }
        });
        for (int n : integerMap.keySet()) {
            priorityQueue.add(n);
            if (priorityQueue.size() > k) {
                priorityQueue.poll();
            }
        }

        int[] intArray = new int[k];
        for (int i = 0; i < k; i++) {
            intArray[i] = priorityQueue.poll();
        }
        return intArray;
    }

//    List<String> stringList=CollectionUtils.

    public static int myAtoi(String str) {

        int len = str.length();
        // str.charAt(i) 方法回去检查下标的合法性，一般先转换成字符数组
        char[] charArray = str.toCharArray();

        // 1、去除前导空格
        int index = 0;
        while (index < len && charArray[index] == ' ') {
            index++;
        }

        // 2、如果已经遍历完成（针对极端用例 "      "）
        if (index == len) {
            return 0;
        }

        // 3、如果出现符号字符，仅第 1 个有效，并记录正负
        int sign = 1;
        char firstChar = charArray[index];
        if (firstChar == '+') {
            index++;
        } else if (firstChar == '-') {
            index++;
            sign = -1;
        }

        // 4、将后续出现的数字字符进行转换
        // 不能使用 long 类型，这是题目说的
        int res = 0;
        while (index < len) {
            char currChar = charArray[index];
            // 4.1 先判断不合法的情况
            if (currChar > '9' || currChar < '0') {
                break;
            }

            // 题目中说：环境只能存储 32 位大小的有符号整数，因此，需要提前判：断乘以 10 以后是否越界
            if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && (currChar - '0') > Integer.MAX_VALUE % 10)) {
                return Integer.MAX_VALUE;
            }
            if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && (currChar - '0') > -(Integer.MIN_VALUE % 10))) {
                return Integer.MIN_VALUE;
            }

            // 4.2 合法的情况下，才考虑转换，每一步都把符号位乘进去
            res = res * 10 + sign * (currChar - '0');
            index++;
        }
        return res;
    }


    /**
     * 打乱数组
     */
    private int[] array;
    private int[] original;
    Random random = new Random();

    public Solution(int[] nums) {
        array = nums;
        original = nums.clone();
    }

    /**
     * Resets the array to its original configuration and return it.
     */
    public int[] reset() {
        array = original;
        original = original.clone();
        return array;
    }

    /**
     * Returns a random shuffling of the array.
     */
    public int[] shuffle() {
        for (int i = 0; i < array.length; i++) {
            compareChange(i, randomInt(array.length, i));
        }
        return array;
    }

    private void compareChange(int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    private int randomInt(int max, int min) {
        return random.nextInt(max - min) + min;
    }

    /**
     * 填充每个节点的下一个右侧节点指针 层序遍历
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {

        if (root == null) {
            return root;
        }
        Node listMost = root;
        while (listMost.left != null) {
            Node head = listMost;
            while (head != null) {
                head.left.next = head.right;
                if (head.next != null) {
                    head.right.next = head.next.left;
                }
                // 遍历直到没有节点
                head = head.next;
            }
            listMost = listMost.left;
        }
        return root;

    }

    /**
     * 生成括号 ， 回溯法
     *
     * @param n
     * @return
     */
    public static List<String> generateParenthesis(int n) {
        List<String> stringList = new LinkedList<>();
        backtrack(stringList, new StringBuilder(), 0, 0, n);
        return stringList;
    }

    public static void backtrack(List<String> ans, StringBuilder sb, int open, int close, int max) {

        if (sb.length() == 2 * max) {
            ans.add(sb.toString());
            return;
        }
        if (open < max) {
            sb.append("(");
            backtrack(ans, sb, open + 1, close, max);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (close < open) {
            sb.append(")");
            backtrack(ans, sb, open, close + 1, max);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * 阶乘的0
     *
     * @param n
     * @return
     */
    public static int trailingZeroes(int n) {

        int zeroCount = 0;
        for (int i = 5; i <= n; i += 5) {
            int powerOf5 = 5;
            while (i % powerOf5 == 0) {
                zeroCount += 1;
                powerOf5 *= 5;
            }
        }
        return zeroCount;
    }

    /**
     * 逆波兰表达式
     *
     * @param tokens
     * @return
     */
    public static int evalRPN(String[] tokens) {

        Stack<Integer> integerStack = new Stack<>();
        Integer operation1, operation2;
        for (String str : tokens) {
            switch (str) {
                case "+":
                    operation1 = integerStack.pop();
                    operation2 = integerStack.pop();
                    integerStack.push(operation1 + operation2);
                    break;
                case "-":
                    operation1 = integerStack.pop();
                    operation2 = integerStack.pop();
                    integerStack.push(operation2 + operation1);
                    break;
                case "*":
                    operation1 = integerStack.pop();
                    operation2 = integerStack.pop();
                    integerStack.push(operation1 * operation2);
                    break;
                case "/":
                    operation1 = integerStack.pop();
                    operation2 = integerStack.pop();
                    integerStack.push(operation2 / operation1);
                    break;
                default:
                    integerStack.push(Integer.parseInt(str));
                    break;
            }
        }
        return integerStack.get(0);

    }

    /**
     * 外观数列, 双指针，递归
     *
     * @param n
     * @return
     */
    public static String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        StringBuilder sb = new StringBuilder();
        String str = countAndSay(n - 1);
        int length = str.length();
        int start = 0;
        for (int i = 1; i < length + 1; i++) {
            if (i == length) {
                sb.append(i - start).append(str.charAt(start));
            } else if (str.charAt(i) != str.charAt(start)) {
                sb.append(i - start).append(str.charAt(start));
                start = i;
            }
        }
        return sb.toString();

    }

    /**
     * 寻找峰值，二分查找，迭代
     *
     * @param nums
     * @return
     */
    public static int findPeakElement(int[] nums) {

        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (right + left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;

    }


    /**
     * 二叉树的锯齿形层次遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        if (root == null) {
            return new LinkedList<>(new ArrayList<>());
        }
        LinkedList<TreeNode> treeNodeQueue = new LinkedList<TreeNode>();
        treeNodeQueue.add(root);
        treeNodeQueue.add(null);
        List<List<Integer>> lists = new LinkedList<>();
        LinkedList<Integer> levelList = new LinkedList<>();
        boolean isListLeft = true;
        while (treeNodeQueue.size() > 0) {
            TreeNode node = treeNodeQueue.pollFirst();
            if (node != null) {
                if (isListLeft) {
                    levelList.addLast(node.val);
                } else {
                    levelList.addFirst(node.val);
                }
                if (node.left != null) {
                    treeNodeQueue.add(node.left);
                }
                if (node.right != null) {
                    treeNodeQueue.add(node.right);
                }
            } else {
                lists.add(levelList);
                levelList = new LinkedList<>();
                if (treeNodeQueue.size() > 0) {
                    treeNodeQueue.add(null);
                    isListLeft = !isListLeft;
                }
            }
        }
        return lists;
    }

    /**
     * 有效的数独
     *
     * @param board
     * @return
     */
    public boolean isValidSudoku(char[][] board) {

        HashMap<Integer, Integer>[] rows = new HashMap[9];
        HashMap<Integer, Integer>[] columns = new HashMap[9];
        HashMap<Integer, Integer>[] boxIndexs = new HashMap[9];
        for (int i = 0; i < 9; i++) {
            rows[i] = new HashMap<>();
            columns[i] = new HashMap<>();
            boxIndexs[i] = new HashMap<>();
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char num = board[i][j];
                if (num != '.') {
                    int boxIndex = (i / 3) * 3 + j / 3;
                    rows[i].put((int) num, rows[i].getOrDefault((int) num, 0) + 1);
                    columns[j].put((int) num, columns[j].getOrDefault((int) num, 0) + 1);
                    boxIndexs[boxIndex].put((int) num, boxIndexs[boxIndex].getOrDefault((int) num, 0) + 1);
                    if (rows[i].get((int) num) > 1 || columns[j].get((int) num) > 1 || boxIndexs[boxIndex].get((int) num) > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * 矩阵置0
     *
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        int rLength = matrix.length, cLength = matrix[0].length;
        int tempValue = 1000000;
        for (int i = 0; i < rLength; i++) {
            for (int j = 0; j < cLength; j++) {
                int value = matrix[i][j];
                if (value == 0) {
                    for (int k = 0; k < rLength; k++) {
                        // 更新行
                        if (matrix[k][j] != 0) {
                            matrix[k][j] = tempValue;
                        }
                    }
                    // 更新列
                    for (int k = 0; k < cLength; k++) {
                        if (matrix[i][k] != 0) {
                            matrix[i][k] = tempValue;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < rLength; i++) {
            for (int j = 0; j < cLength; j++) {
                if (matrix[i][j] == tempValue) {
                    matrix[i][j] = 0;
                }
            }
        }
    }


    /**
     * 遍历整个矩阵，如果 cell[i][j] == 0 就将第 i 行和第 j 列的第一个元素标记。
     * 第一行和第一列的标记是相同的，都是 cell[0][0]，所以需要一个额外的变量告知第一列是否被标记，同时用 cell[0][0] 继续表示第一行的标记。
     * 然后，从第二行第二列的元素开始遍历，如果第 r 行或者第 c 列被标记了，那么就将 cell[r][c] 设为 0。这里第一行和第一列的作用就相当于方法一中的 row_set 和 column_set 。
     * 然后我们检查是否 cell[0][0] == 0 ，如果是则赋值第一行的元素为零。
     * 然后检查第一列是否被标记，如果是则赋值第一列的元素为零。
     *
     * @param matrix
     */
    public void setZeroes1(int[][] matrix) {
        int rLength = matrix.length, cLength = matrix[0].length;
        boolean isCol = false;
        for (int i = 0; i < rLength; i++) {
            if (matrix[i][0] == 0) {
                isCol = true;
            }
            for (int j = 1; j < cLength; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < rLength; i++) {
            for (int j = 1; j < cLength; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (matrix[0][0] == 0) {
            for (int i = 0; i < cLength; i++) {
                matrix[0][i] = 0;
            }
        }
        if (isCol) {
            for (int i = 0; i < rLength; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    /**
     * 单词接龙
     * 对给定的 wordList 做预处理，找出所有的通用状态。将通用状态记录在字典中，键是通用状态，值是所有具有通用状态的单词。
     * 将包含 beginWord 和 1 的元组放入队列中，1 代表节点的层次。我们需要返回 endWord 的层次也就是从 beginWord 出发的最短距离。
     * 为了防止出现环，使用访问数组记录。
     * 当队列中有元素的时候，取出第一个元素，记为 current_word。
     * 找到 current_word 的所有通用状态，并检查这些通用状态是否存在其它单词的映射，这一步通过检查 all_combo_dict 来实现。
     * 从 all_combo_dict 获得的所有单词，都和 current_word 共有一个通用状态，所以都和 current_word 相连，因此将他们加入到队列中。
     * 对于新获得的所有单词，向队列中加入元素 (word, level + 1) 其中 level 是 current_word 的层次。
     * 最终当到达期望的单词，对应的层次就是最短变换序列的长度
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {

        if (!wordList.contains(endWord)) {
            return 0;
        }
        int length = beginWord.length();
        Map<String, List<String>> stringListMap = new HashMap<>();
        wordList.forEach(word -> {
            for (int i = 0; i < length; i++) {
                String newWord = word.substring(0, i) + "*" + word.substring(i + 1, length);
                List<String> stringList = stringListMap.getOrDefault(newWord, new ArrayList<>());
                stringList.add(word);
                stringListMap.put(newWord, stringList);
            }
        });
        Queue<Pair<String, Integer>> queue = new LinkedList<>();
        queue.add(new Pair<>(beginWord, 1));
        // 记录该单词是否被访问过
        Set<String> visitedSet = new HashSet<>();
        while (!queue.isEmpty()) {
            Pair<String, Integer> node = queue.remove();
            String currWord = node.getKey();
            int level = node.getValue();
            for (int i = 0; i < length; i++) {
                String newWord = currWord.substring(0, i) + "*" + currWord.substring(i + 1, length);
                for (String word : stringListMap.getOrDefault(newWord, new ArrayList<>())) {
                    if (word.equals(endWord)) {
                        return level + 1;
                    }
                    if (!visitedSet.contains(word)) {
                        visitedSet.add(word);
                        queue.add(new Pair<>(word, level + 1));
                    }
                }
            }
        }
        return 0;
    }

    public double quickMul(double x, long n) {

        if (n == 0) {
            return 1.0;
        }
        double y = quickMul(x, n / 2);
        return n % 2 == 0 ? y * y : y * y * x;
    }

    /**
     * Pow(x, n)
     * 计算x的n次幂时，递归计算x的n/2次幂，结果向下取整，假设计算结果为y,当n为偶数时，x的n次幂等于y*y，反之，x的n次幂等于y*y*x;
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        return n > 0 ? quickMul(x, n) : 1.0 / quickMul(x, -n);
    }


    /**
     * 快速幂+迭代
     *
     * @param x
     * @param n
     * @return
     */
    private double quickMul(double x, int n) {
        double ans = 1.00;
        // 贡献的初始值为 x
        double xContri = x;
        while (n > 0) {
            if (n % 2 == 1) {
                // 如果 N 二进制表示的最低位为 1，那么需要计入贡献
                ans *= xContri;
            }
            // 将贡献不断地平方
            xContri *= xContri;
            // 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
            n /= 2;
        }
        return ans;
    }

    /**
     * 使用排序合并区间
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {

        // 升序排序
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        List<int[]> merged = new LinkedList<>();
        for (int i = 0; i < intervals.length; i++) {
            int left = intervals[i][0], right = intervals[i][1];
            // 所有能够合并的区间都必然是连续的 a[i].end<a[j].start≤a[j].end<a[k].start,
            if (merged.size() == 0 || left > merged.get(merged.size() - 1)[1]) {
                merged.add(new int[]{left, right});
            } else {
                // 右区间，替换为较大值
                merged.get(merged.size() - 1)[1] = (Math.max(right, merged.get(merged.size() - 1)[1]));
            }
        }
        return merged.toArray(new int[merged.size()][]);

    }

    class LRUCache {

        private HashMap<Integer, DbLinkedNode> hashMap = new HashMap<>();
        int size;
        int capacity;
        private DbLinkedNode head, tail;

        class DbLinkedNode {
            int value;
            int key;
            DbLinkedNode pre;
            DbLinkedNode next;

            public DbLinkedNode() {

            }

            public DbLinkedNode(int value, int key) {
                this.value = value;
                this.key = key;
            }
        }

        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            head = new DbLinkedNode();
            tail = new DbLinkedNode();
            head.next = tail;
            tail.pre = head;
        }

        public int get(int key) {
            DbLinkedNode node = hashMap.get(key);
            if (node == null) {
                return -1;
            }
            // 如果 key 存在，先通过哈希表定位，再移到头部
            moveToHead(node);
            return node.value;
        }

        /**
         * 重点
         * @param key
         * @param value
         */
        public void put(int key, int value) {
            DbLinkedNode node = hashMap.get(key);
            if (node == null) {
                DbLinkedNode newNode = new DbLinkedNode(key, value);
                addtoHead(newNode);
                hashMap.put(key, newNode);
                size++;
                if (size > capacity) {
                    DbLinkedNode tail = removeTail();
                    hashMap.remove(tail.key);
                    size--;
                }
            } else {
                // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
                node.value = value;
                moveToHead(node);
            }
        }

        /**
         * 重点
         * @param node
         */
        private void addtoHead(DbLinkedNode node) {
            node.pre = head;
            node.next = head.next;
            head.next.pre = node;
            head.next = node;
        }

        private void removeNode(DbLinkedNode node) {
            node.pre.next = node.next;
            node.next.pre = node.pre;
        }

        private void moveToHead(DbLinkedNode node) {
            removeNode(node);
            addtoHead(node);
        }

        private DbLinkedNode removeTail() {
            DbLinkedNode res = tail.pre;
            removeNode(res);
            return res;
        }

    }


    public static void main(String[] args) {
        //  String s = "a";
        int[] nums = {1, 2, 3, 1};
//        int[] arrays = topKFrequent(nums, 2);
//        String s = "asbnb";
//        long start = System.currentTimeMillis();
//        System.out.println(findPeakElement(nums));
//        long end = System.currentTimeMillis();
//        System.out.println("耗时" + (end - start));
        //  System.out.println(longestPalindrome(s));
        // System.out.println(stringList.toString());
        //System.out.println(titleToNumber("B"));
//        Arrays.sort(nums);
//        System.out.println(nums);
        //  String s = "anagram";
        // String t = "nagaram";
        // bubbleSort(nums);
//        quickSort(nums, 0, nums.length - 1);
//        System.out.println(Arrays.toString(nums));
//        String str = "0003320F3298182||01|051440202065|TX09|61973669|882005251708423112|4200000533202005252079547943|1.00|1|SUCCESS|2020-05-25 17:08:47|20200525||0.00|1026100000020";
//        String[] strings = StringUtils.split(str, "|");
//        System.out.println(Arrays.toString(strings));
        // System.out.println(isAnagram(s, t));
        //System.out.println(rob(nums));

        // System.out.println(partition(s));
//        //int result = searchInsert(nums, 2);
//        //List<Integer> result = getRow(3);
//        int[] result = exchange(nums);
//        for (int i = 0; i < result.length; i++) {
//            System.out.println(result[i]);
//        }
//        int n = 100;
//        int result = sumNums(n);
//        System.out.println(result);
        //System.out.println(singleNumber(nums));
//        List<List<Integer>> integerList = permute(nums);
//        System.out.println(integerList);
        //int result = lengthOfLastWord1(s);
//        List<List<String>> listList = groupAnagrams(nums);
//        System.out.println(listList);
//        long start = System.currentTimeMillis();
//        int result = countPrimes(count);
//        long end = System.currentTimeMillis();
//        System.out.println("耗时:" + (end - start));
//        System.out.println(result);
//        String string = "23484";
//        int [] ints={0,4,5,7,0,8};
//        //List<String> str = letterCombinations(string);
//        //System.out.println("结果" + str);
//        moveZeroes(ints);
        // Date date=getYesterday();
//        String date = "20190802";
//        SimpleDateFormat sb = new SimpleDateFormat("yyyyMMdd");
//        try {
//            Date date1 = sb.parse(date);
//            System.out.println(date1);
//        } catch (ParseException e) {
//            e.printStackTrace();
//        }
//        System.out.println("启动计数线程");
//        System.out.println("计数值：" + latch.getCount());
//        getThread().start();
//        while (latch.getCount() != 0) {
//            latch.countDown();
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            System.out.println("计数值：" + latch.getCount());
//        }
    }


}
