package org.iplatform.microservices.brain.hole.bean;

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
