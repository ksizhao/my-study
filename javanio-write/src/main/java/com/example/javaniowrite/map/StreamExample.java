package com.example.javaniowrite.map;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author zhaoliancan
 * @version 1.0.0
 * @ClassName StreamExample.java
 * @Description TODO
 * @createTime 2019年09月10日 17:15:00
 */
public class StreamExample {

    public static void main(String[] args) {
        Trader raoul = new Trader("Raoul", "Cambridge");
        Trader mario = new Trader("Mario", "Milan");
        Trader alan = new Trader("Alan", "Cambridge");
        Trader brian = new Trader("Brian", "Cambridge");

        List<Transaction> transactions = Arrays.asList(
                new Transaction(brian, 2011, 300),
                new Transaction(raoul, 2012, 1000),
                new Transaction(raoul, 2011, 400),
                new Transaction(mario, 2012, 710),
                new Transaction(mario, 2012, 700),
                new Transaction(alan, 2012, 950)
        );

        List<Transaction> list = transactions.stream().filter(transaction -> transaction.getYear() > 2011).sorted(Comparator.comparing(Transaction::getValue)).collect(Collectors.toList());
        //打印所有城市在剑桥的交易员的交易额
        transactions.stream().filter(transaction -> transaction.getTrader().getCity().equals("Cambridge")).map(Transaction::getValue).sorted().forEach(System.out::println);
        // 将所有的交易按照年份分组，存放在一个Map中
        //Map<Integer,Transaction> map=list.stream().collect(Collectors.toMap(Transaction::getYear,transaction -> transaction));
        Map<Integer, Transaction> yearMap = transactions.stream()
                .collect(Collectors.toMap(Transaction::getYear, transaction -> transaction));
       // System.out.println(map.size());
        System.out.println(yearMap.size());

    }
}
