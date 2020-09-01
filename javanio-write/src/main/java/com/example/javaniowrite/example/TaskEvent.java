package com.example.javaniowrite.example;

import org.springframework.context.ApplicationEvent;

/**
 * @author zhaoliancan
 * @description 自定义事件
 * @create 2019-05-13 19:43
 */
public class TaskEvent  extends ApplicationEvent {

    public TaskEvent(Object source) {
        super(source);
    }
}
