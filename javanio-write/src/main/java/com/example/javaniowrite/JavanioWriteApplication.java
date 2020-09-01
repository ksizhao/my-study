package com.example.javaniowrite;

import com.example.javaniowrite.example.TaskEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.redis.core.StringRedisTemplate;

@SpringBootApplication
public class JavanioWriteApplication {

    @Autowired
   private  StringRedisTemplate redisTemplate;



    public static void main(String[] args) {

        SpringApplication application=new SpringApplication(JavanioWriteApplication.class);

        ConfigurableApplicationContext context=application.run(args);
        // 发布事件
        context.publishEvent(new TaskEvent(new Object()));
        context.close();
        //SpringApplication.run(JavanioWriteApplication.class, args);
    }



}

