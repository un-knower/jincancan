package org.iplatform.microservices.brain.ui.controller;

import java.security.Principal;
import java.util.Map;

import org.iplatform.microservices.brain.ui.feign.TestClient;
import org.iplatform.microservices.core.http.RestResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

/**
 * @author zhanglei
 */
@Controller
@RequestMapping("/test")
public class TestController  {
	private static final Logger LOG = LoggerFactory.getLogger(TestController.class);
	
	@Autowired
	TestClient testClient;
	
	@RequestMapping("/topo")
	public String index(ModelMap map,Principal principal) throws Exception {	
		return "test/topo/topo";		
	}
	
	@RequestMapping("/charts")
	public String charts(ModelMap map,Principal principal) throws Exception {	
		return "test/charts/charts";		
	}
		
	
	@RequestMapping("/service")
	public String service(ModelMap map,Principal principal) throws Exception {	
		return "test/service/service";		
	}	

	@RequestMapping("/testservice")
	public ResponseEntity<RestResponse<Map>> testservice(ModelMap map,@RequestParam("param") String param, Principal principal) throws Exception {	
		return testClient.testmethod(param);		
	}	
	
}