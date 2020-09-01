package com.example.javaniowrite.example;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashMap;

/**
 * @author zhaoliancan
 * @description 邮件监听器
 * @create 2019-05-13 19:45
 */
@Component
public class MailListener  implements ApplicationListener<TaskEvent> {

    private String eMail="17816873148@163.com";

//    @Autowired
//    TaskEvent taskEvent;
//
//    public  MailListener(TaskEvent taskEvent) {
//
//        this.taskEvent=taskEvent;
//
//    }

    @Override
    public void onApplicationEvent(TaskEvent event) {
        //System.out.println("任务事件:"+event.getSource()+"触发邮件通知监听器，将发送邮件到:"+eMail);
        System.out.println("接收到事件"+event.getClass());
    }
}
