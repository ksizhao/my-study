package com.example.javaniowrite;

import com.example.javaniowrite.IO.ReadFile;
import com.example.javaniowrite.designModule.state.Action;
import com.example.javaniowrite.designModule.state.EntranceMachine;
import com.example.javaniowrite.designModule.state.impl.LockedEntranceMachineState;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.BDDAssertions.then;

@RunWith(SpringRunner.class)
@SpringBootTest
public class JavanioWriteApplicationTests {

    @Autowired
    private StringRedisTemplate redisTemplate;

    @Test
    public void contextLoads() {

        Calendar startTime = Calendar.getInstance();
        List<String> stringList = new LinkedList<>();
        for (int i = 0; i < 300000; i++) {
            stringList.add("这是第" + i + "行");
        }
        batchCacheInfo("zlc", stringList, 20);
        Calendar redisTime = Calendar.getInstance();
        System.out.println("本次写redis用时" + (redisTime.getTimeInMillis() - startTime.getTimeInMillis()) / 1000 + "秒" + (redisTime.getTimeInMillis() - startTime.getTimeInMillis()) % 1000 + "毫秒");


//            WriteFile writeFile=new WriteFile(redisTemplate);
//            writeFile.start();


        ReadFile readFile = new ReadFile();
        // readFile.start();

    }

    private void batchCacheInfo(final String key, final List<String> dataList, final long time) {
        //使用pipeline方式
        final List<Object> objects = redisTemplate.executePipelined(new RedisCallback<List<Object>>() {
            @Override
            public List<Object> doInRedis(RedisConnection connection) throws DataAccessException {
                connection.openPipeline();
                for (String str : dataList) {
                    connection.lPush(key.getBytes(), str.getBytes());
                    // 批量设置过期时间
                    //connection.expire(key.getBytes(),3600*time);
                }
                return null;
            }
        });
    }

    @Test
    public void test() {

        AtomicInteger atomicInteger = new AtomicInteger(0);
        final int num = 100;

        for (int i = 0; i < num; i++) {

            new Thread(new Runnable() {
                @Override
                public void run() {
                    atomicInteger.getAndIncrement();
                }
            }).start();
        }
        System.out.println("计数值" + atomicInteger);


    }

    @Test
    public void should_be_unlocked_when_insert_coin_given_a_entrance_machine_with_locked_state() {
        EntranceMachine entranceMachine = new EntranceMachine(new LockedEntranceMachineState());

        String result = entranceMachine.execute(Action.INSERT_COIN);

        then(result).isEqualTo("opened");
        then(entranceMachine.isUnlocked()).isTrue();
    }


}

