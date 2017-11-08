package org.iplatform.microservices.brain.service.bean;

import java.io.Serializable;

public class TestBean implements Serializable  {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
