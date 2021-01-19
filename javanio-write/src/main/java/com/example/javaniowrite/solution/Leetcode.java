package com.example.javaniowrite.solution;

import java.util.HashMap;

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
     * 复制带随机指针的节点
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        if(vistedMap.containsKey(head)){
            return vistedMap.get(head);
        }
        Node node = new Node(head.val, null, null);
        vistedMap.put(head, node);
        node.next = copyRandomList(head.next);
        node.random = copyRandomList(head.random);
        return node;

    }
}
