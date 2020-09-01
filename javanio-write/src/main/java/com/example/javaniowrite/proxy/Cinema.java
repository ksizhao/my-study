package com.example.javaniowrite.proxy;

/**
 * @author zhaoliancan
 * @description 电影院
 * @create 2019-01-03 19:11
 */
public class Cinema implements Movie {

    RealMovie realMovie;


     public Cinema(RealMovie realMovie) {
         super();
         this.realMovie=realMovie;

     }

    @Override
    public void play() {
        realMovie.play();
        System.out.println("你正在电影院看电影");
        isPlay(true);

    }


    public void isPlay(boolean var) {

        if(var) {

            System.out.println("电影正准备开始播放，卖爆米花");
        }else {

            System.out.println("电影已经播放完毕，请把爆米花带回家");
        }
    }

}
