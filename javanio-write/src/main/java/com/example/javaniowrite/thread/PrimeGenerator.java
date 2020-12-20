package com.example.javaniowrite.thread;

import javax.annotation.concurrent.ThreadSafe;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

@ThreadSafe
public class PrimeGenerator {


    static class PrimeGenerators implements Runnable {
        private final List<BigInteger> bigIntegers = new ArrayList<>();

        private volatile boolean cancelFlag;

        public void run() {
            BigInteger one = BigInteger.ONE;
            while (!cancelFlag) {
                one = one.nextProbablePrime();
                synchronized (this) {
                    bigIntegers.add(one);
                }
            }
        }

        public void cancel() {
            cancelFlag = true;
        }

        public synchronized List<BigInteger> getLIst() {
            return new ArrayList<>(bigIntegers);
        }
    }

    public static void main(String[] args) {
        System.out.println(get().toString());

    }

    private static  List<BigInteger> get() {
        PrimeGenerators primeGenerator = new PrimeGenerators();
        new Thread((Runnable) primeGenerator).start();
        try {
             Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            primeGenerator.cancel();
        }
        return primeGenerator.getLIst();
    }

}
