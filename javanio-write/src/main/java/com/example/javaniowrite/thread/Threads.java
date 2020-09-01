package com.example.javaniowrite.thread;

/**
 * @author zhaoliancan
 * @description 线程类
 * @create 2019-02-20 14:48
 */
public class Threads {


    private static User u=new User();



    public static void main(String[] args) throws InterruptedException {
//      Thread t=new Thread() {
//          @Override
//          public void run() {
//             // super.run();
//              while (true) {
//                  if(Thread.currentThread().isInterrupted()) {
//                      System.out.println("Interrupted");
//                      break;
//                  }
//                  Thread.yield();
//              }
//
//          }
//      };
//      t.start();
//      Thread.sleep(100);
//      t.interrupt();
        Thread [] threads=new Thread[100];
        for (int i=0;i<100;i++) {
            threads[i]=new volatileThread();
            threads[i].start();
        }

//        for (int i=0;i<200;i++) {
//            threads[i].start();
//        }

    }

    private static class User{
        private int id;
        private String name;

        public User () {
            id=0;
            name="0";

        }

        public int getId() {
            return id;
        }

        public void setId(int id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{" +
                    "id=" + id +
                    ", name='" + name + '\'' +
                    '}';
        }
    }


    private static class WriteThred extends Thread {

        @Override
        public void run() {
            while (true) {
                synchronized (u) {
                    int value= (int) (System.currentTimeMillis()/1000);
                    u.setId(value);
                }

                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                Thread.yield();
            }
        }
    }


    private static class ReadThread extends Thread {

        @Override
        public void run() {
           while (true) {
               synchronized (u) {
                   if (u.getId()!=Integer.parseInt(u.getName())) {
                       System.out.println(u.toString());
                   }

               }
           }
        }

    }

    private static  class volatileThread extends Thread {
       volatile private static int count=0;

        @Override
        public void run() {
            accountC();
        }

        private static void accountC() {
            for (int i=0;i<100;i++) {
                count=i;
            }

            System.out.println("count="+count);
        }


    }





}
