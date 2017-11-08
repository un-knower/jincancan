package org.iplatform.microservices.brain.hole.service.dao;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.iplatform.microservices.brain.hole.service.domain.TestDO;

@Mapper
public interface TestMapper {
	
	@Select("select * from empty_test")
	List<TestDO> getAll();
}
