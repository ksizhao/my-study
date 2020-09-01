package com.example.javaniowrite.socket;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * @author zhaoliancan
 * @description 服务端
 * @create 2019-08-01 9:32
 */
@Slf4j
public class SocketServer {


    public static void main(String[] args) {

        try {
            ServerSocket serverSocket=new ServerSocket(8090);
            ThreadPoolExecutor executor=getPool();

            while (true) {
                Socket socket= serverSocket.accept();
                Runnable runnable=()->{
                        try {
                            log.info("客户端信息{}",socket.getLocalAddress());
                            BufferedReader bufferedReader=new BufferedReader(new InputStreamReader(socket.getInputStream()));
                            String message=null;
                            while ((message=bufferedReader.readLine())!=null) {

                                log.info("从客户端读取内容={}",message);
                            }
                            socket.shutdownInput();
                            BufferedWriter bufferedWriter=new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
                            bufferedWriter.write("我已收到消息");
                            bufferedWriter.flush();
                            socket.shutdownOutput();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                };
                executor.submit(runnable);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static ThreadPoolExecutor getPool() {
        ThreadFactory namedThreadFactory = new ThreadFactoryBuilder()
                .setNameFormat("demo-pool-%d").build();
        return new ThreadPoolExecutor(5, 11,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<Runnable>(1024), namedThreadFactory, new ThreadPoolExecutor.AbortPolicy());
    }
}
