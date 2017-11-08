package org.iplatform.microservices.brain.ui.feign;

import java.util.Map;

import org.iplatform.microservices.core.http.RestResponse;
import org.springframework.cloud.netflix.feign.FeignClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;

@FeignClient("empty-service")
public interface TestClient {

	@RequestMapping(value = "emptyservice/api/v1/test/testmethod", method = RequestMethod.GET)
	public ResponseEntity<RestResponse<Map>> testmethod(@RequestParam(value = "param") String param);    
}
