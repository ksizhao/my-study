package com.example.javaniowrite.map;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author zhaoliancan
 * @version 1.0.0
 * @ClassName Transaction.java
 * @Description TODO
 * @createTime 2019年09月10日 17:15:00
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Transaction {
    private Trader trader;
    private int year;
    private int value;
}
