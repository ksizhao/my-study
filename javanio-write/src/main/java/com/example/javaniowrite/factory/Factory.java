package com.example.javaniowrite.factory;

public interface Factory {
    /**
     * 获取动物实例
     * @return
     */
    public Animal getAnimal();

    /**
     * 动物对应能力
     * @return
     */
    public Action getAction();
}
