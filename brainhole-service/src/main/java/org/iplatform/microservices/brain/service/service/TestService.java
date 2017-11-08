package org.iplatform.microservices.brain.service.service;

import java.io.File;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.security.Principal;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;

import javax.annotation.PostConstruct;
import javax.servlet.http.HttpServletRequest;

import org.apache.activemq.ActiveMQConnectionFactory;
import org.iplatform.microservices.brain.service.bean.TestBean;
import org.iplatform.microservices.brain.service.service.dao.TestMapper;
import org.iplatform.microservices.brain.service.service.domain.TestDO;
import org.iplatform.microservices.core.dfss.DFSSClient;
import org.iplatform.microservices.core.dfss.DfssFileDO;
import org.iplatform.microservices.core.global.GlobalInterceptorConstant;
import org.iplatform.microservices.core.http.RestResponse;
import org.iplatform.microservices.core.scheduled.lock.core.LockModel;
import org.iplatform.microservices.core.scheduled.lock.core.SchedulerLock;
import org.iplatform.microservices.core.security.UserDetails;
import org.iplatform.microservices.core.security.UserDetailsUtil;
import org.iplatform.microservices.core.web.WebListener;
import org.iplatform.microservices.lns.ServiceLic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.CacheManager;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.jms.core.JmsTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;

/**
 * @author zhanglei
 *
 */
@Configuration
@Service
@RestController
@RequestMapping("/api/v1/test")
public class TestService {
	private static final Logger logger = LoggerFactory.getLogger(TestService.class);

	@Autowired
	private CacheManager cacheManager;

	@Autowired
	ActiveMQConnectionFactory activemqConnectionFactory;

	@Autowired
	ScheduledThreadPoolExecutor scheduledThreadPoolExecutor;

	@Autowired
	JmsTemplate queueJmsTemplate;
	
	@Autowired
	UserDetailsUtil userDetailsUtil;	
	
	@Autowired
	TestMapper testMapper;
	
	@Autowired
	DFSSClient dfssClient;

	/**
	 * 初始化加载所有策略
	 */
	@PostConstruct
	public void init() {
		logger.info("类实例化");
	}
	
    @ServiceLic
    @RequestMapping(value = "/hello", method = RequestMethod.GET)
    public void hello(){        
        logger.info("test");
    }
    
	@RequestMapping(value = "/testmethod", method = RequestMethod.GET)
	@ApiOperation(value = "测试方法", notes = "此方法只用于测试服务能力")
	public ResponseEntity<RestResponse<Map>> testmethod(
			@ApiParam(value = "参数", required = true)
			@RequestParam(value = "param") String param,Principal principal,HttpServletRequest request){
	    logger.info(request.getHeader("x-ipf-pageid"));
	    List<String> headers = new ArrayList();
        Enumeration<String> headenum = request.getHeaderNames();
        while (headenum.hasMoreElements()) {
            String headerName = headenum.nextElement();
            if (headerName.startsWith(GlobalInterceptorConstant.XIPF_HEADER_PREFIX)) {
                String headerValue = request.getHeader(headerName);
                headers.add(headerName +"="+headerValue);
            }
        } 
        
		RestResponse<Map> response = new RestResponse<Map>();
		Map data = new HashMap();
		
    	UserDetails userDetails = new UserDetails();
		userDetails.setUsername(principal.getName());
		userDetails = userDetailsUtil.getUserDetails(userDetails);
		
		data.put("param", param);
		data.put("userDetails", userDetails);
		data.put("testBean", new TestBean());		
		data.put("serviceId", WebListener.webport);
		data.put("X-IPF", headers);
		
		try {
            String host = InetAddress.getLocalHost().getHostAddress();
            data.put("host", host);
        } catch (UnknownHostException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        response.setData(data);
        response.setSuccess(Boolean.TRUE);
        return new ResponseEntity<>(response, HttpStatus.OK);			
	}
	
	@RequestMapping(value = "/testmethodwithdb", method = RequestMethod.GET)
	
	public ResponseEntity<RestResponse<List<TestDO>>> testmethodWithDB(){
        RestResponse<List<TestDO>> response = new RestResponse();    
	    List<TestDO> dbos = testMapper.getAll();
        response.setData(dbos);
        response.setSuccess(Boolean.TRUE);
        return new ResponseEntity<>(response, HttpStatus.OK);
	}
	
    @RequestMapping(value = "/testdfssupload", method = RequestMethod.GET)
    public @ResponseBody DfssFileDO testdfssupload() throws Exception{   
        String fileId = this.dfssClient.add("开发手册.pdf", new File("/Users/zhanglei/Desktop/开发手册.pdf"));
        this.dfssClient.get(fileId, "/Users/zhanglei/Desktop/下载.pdf");
        
        this.dfssClient.update(fileId, "ssoclient.zip", new File("/Users/zhanglei/Desktop/ssoclient.zip"));
        
        DfssFileDO dfssinfo = this.dfssClient.info(fileId);
        this.dfssClient.delete(fileId);
        return dfssinfo;
    }	
    
    @Scheduled(fixedDelay = 5000)
    @SchedulerLock(model=LockModel.iammaster)
    public void fixedRate() {
        System.out.println("每隔5秒执行一次");
    }
}
