package org.iplatform.microservices.brain.ui.controller;

import java.security.Principal;

import org.iplatform.microservices.core.security.UserDetailsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * @author zhanglei
 */
@Controller
public class IndexController {
	private static final Logger LOG = LoggerFactory.getLogger(IndexController.class);
	
	@Autowired
	private UserDetailsUtil userDetailsUtil;
	
	@RequestMapping("/")
	public String index(ModelMap map,Principal principal) throws Exception {	
		//每次进入首页都清除用户缓存信息，以便于后续操作会重新从认证服务器中获取后再次缓存
		userDetailsUtil.removeUserDetails(principal.getName());		
		return "index";		
	}	
	
    @RequestMapping("/systemmanager")
    public String user(ModelMap map) throws Exception {
        map.put("sidebar", "systemmanager");
        return "system";
    }	
	
}