package org.iplatform.microservices.brain.service;

import org.iplatform.microservices.service.IPlatformServiceApplication;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.EnableAspectJAutoProxy;
import org.springframework.jms.annotation.EnableJms;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableOAuth2Client;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@EnableOAuth2Client
@SpringBootApplication
@EnableTransactionManagement
@EnableDiscoveryClient
@EnableEurekaClient
@EnableResourceServer
@EnableJms
@EnableCaching
@ComponentScan(basePackages = "org.iplatform.microservices,org.iplatform.microservices.brain.service")
@EnableAspectJAutoProxy
public class BrainServiceApplication extends IPlatformServiceApplication {
    
    private static final Logger LOG = LoggerFactory.getLogger(BrainServiceApplication.class);
    
    public static void main(String[] args) throws Exception {
        run(BrainServiceApplication.class, args);
    }
}
