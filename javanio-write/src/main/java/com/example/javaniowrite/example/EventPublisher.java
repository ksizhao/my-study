package com.example.javaniowrite.example;

import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.ApplicationEvent;
import org.springframework.stereotype.Component;

/**
 * @author zhaoliancan
 * @description 事件发布器
 * @create 2019-05-13 19:48
 */
@Component
public class EventPublisher  implements ApplicationContextAware {

    private ApplicationContext applicationContext;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {

        this.applicationContext=applicationContext;

    }


    /**
     * 发布事件
     * @param event
     */
    public  void publishEvent(ApplicationEvent event) {
        applicationContext.publishEvent(event);
    }
}
