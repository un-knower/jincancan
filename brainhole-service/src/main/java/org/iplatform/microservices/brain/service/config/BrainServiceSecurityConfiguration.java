package org.iplatform.microservices.brain.service.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@Configuration
public class BrainServiceSecurityConfiguration extends WebSecurityConfigurerAdapter {

    //设置认证不拦截规则
	@Override
	public void configure(WebSecurity web) throws Exception {
	    //自定义跳过认证拦截的路径
	    web.ignoring().antMatchers("/api/v1/test/**");
	}

}
