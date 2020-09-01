package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 狗工厂
 * @create 2019-02-12 17:02
 */
public class DogFactory  implements Factory{

    

    @Override
    public Animal getAnimal() {
        return new Dog();
    }


    @Override
    public Action getAction() {

        return new DogAction();
    }
}
