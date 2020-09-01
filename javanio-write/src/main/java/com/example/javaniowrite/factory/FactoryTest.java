package com.example.javaniowrite.factory;

/**
 * @author zhaoliancan
 * @description 工厂测试类
 * @create 2019-02-12 17:04
 */
public class FactoryTest {

    public static void main(String[] args) {
//        // 实例化工厂
//        Factory factory=new CutFactory();
//
//        // 实例化产品
//        Animal cut=factory.getAnimal();
//        cut.getAnimalType();
//
//        Factory factory1=new DogFactory();
//        Animal  dog=factory1.getAnimal();
//        dog.getAnimalType();

        Factory factory;
        Action action;
        Animal animal;

        factory=new CutFactory();
        // 获取动物种类
        animal=factory.getAnimal();
        animal.getAnimalType();

        // 获取动物对应的行为
        action=factory.getAction();
        action.behaviour();


    }
}
