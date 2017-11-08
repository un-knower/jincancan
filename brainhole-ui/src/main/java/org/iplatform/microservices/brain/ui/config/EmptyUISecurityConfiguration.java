package org.iplatform.microservices.brain.ui.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

/**
 * @author zhanglei
 * Spring Security 配置
 */
@Configuration
@EnableWebSecurity
@Order(102)
public class EmptyUISecurityConfiguration extends WebSecurityConfigurerAdapter {

	//设置认证不拦截规则
	@Override
	public void configure(WebSecurity web) throws Exception {
        //自定义跳过认证拦截的路径
        //web.ignoring().antMatchers("/xx").antMatchers("/xxx");		
	}

}
