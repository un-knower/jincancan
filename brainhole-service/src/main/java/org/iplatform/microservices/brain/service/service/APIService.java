package org.iplatform.microservices.brain.service.service;

import javax.annotation.PostConstruct;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.CacheManager;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author zhanglei
 *
 */
@Configuration
@Service
@RestController
@RequestMapping("/api/v1")
public class APIService {
	private static final Logger logger = LoggerFactory.getLogger(APIService.class);

	@Autowired
	private CacheManager cacheManager;

	@PostConstruct
	public void init() {
		logger.info("类实例化");
	}
	
    @RequestMapping(value = "/hi", method = RequestMethod.GET)
    @ResponseBody
    public String hello(){        
        return "Hi I'm Brain Hole";
    }
}
