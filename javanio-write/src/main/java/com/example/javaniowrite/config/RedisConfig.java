package com.example.javaniowrite.config;

import com.sun.org.apache.bcel.internal.classfile.Unknown;
import org.omg.CORBA.portable.UnknownException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.jedis.JedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import redis.clients.jedis.JedisPoolConfig;

/**
 * @author zhaoliancan
 * @description redis配置类
 * @create 2019-01-08 10:23
 */

@Configuration
public class RedisConfig {

//    @Value("${spring.redis.host}")
//    private String recordRedisHost;
//
//    @Value("${spring.redis.port}")
//    private Integer recordRedisPort;
//
//    @Value("${spring.redis.pool.max-active}")
//    private Integer recordRedisMax;
//
//
//    @Value("${server.redis.host}")
//    private String serverRedisHost;
//
//    @Value("${server.redis.port}")
//    private Integer serverRedisPort;
//
//
//
//    @Bean(name = "recordRedisTemplate")
//    public StringRedisTemplate recordRedisTemplate()  throws UnknownException {
//        StringRedisTemplate redisTemplate=new StringRedisTemplate();
//        redisTemplate.setConnectionFactory(connectionFactory(recordRedisHost,recordRedisPort,recordRedisMax));
//
//        return  redisTemplate;
//    }
//
//
//    @Bean(name = "serverRedisTemplete")
//    public  StringRedisTemplate serverRedisTemplete() throws UnknownException {
//
//        StringRedisTemplate redisTemplate=new StringRedisTemplate();
//        redisTemplate.setConnectionFactory(connectionFactory(serverRedisHost,serverRedisPort,recordRedisMax));
//        return  redisTemplate;
//    }
//
//
//
//    private JedisConnectionFactory connectionFactory(String host,Integer port,Integer maxActive) {
//
//        JedisConnectionFactory jedisConnectionFactory=new JedisConnectionFactory(new JedisPoolConfig());
//        JedisPoolConfig jedisPoolConfig=new JedisPoolConfig();
//        jedisPoolConfig.setMaxTotal(maxActive);
//        jedisConnectionFactory.setPoolConfig(jedisPoolConfig);
//        jedisConnectionFactory.afterPropertiesSet();
//
//        return  jedisConnectionFactory;
//    }


}
