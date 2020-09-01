package com.example.javaniowrite.socket;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.Socket;

/**
 * @author zhaoliancan
 * @description 客户端
 * @create 2019-08-01 9:32
 */
@Slf4j
public class SocketClient {

    public static void main(String[] args) {
        try {
            for (int i = 0; i < 20; i++) {
                Socket socket=new Socket("localhost",8090);
                BufferedWriter bufferedWriter=new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
                String message="服务器你好，我是客户端";
                bufferedWriter.write(message);
                log.info("客户端发送消息");
                bufferedWriter.flush();
                socket.shutdownOutput();

                BufferedReader bufferedReader=new BufferedReader(new InputStreamReader(socket.getInputStream()));
                String string=bufferedReader.readLine();
                log.info("收到服务端消息={}",string);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
