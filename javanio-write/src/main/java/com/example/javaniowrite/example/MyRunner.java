package com.example.javaniowrite.example;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

/**
 * @author zhaoliancan
 * @description
 * @create 2019-07-04 17:21
 */
@Component
@Order(1)

public class MyRunner implements ApplicationRunner {

    @Override
    public void run(ApplicationArguments args) throws Exception {
        System.out.println("自定义开始");
    }
}
