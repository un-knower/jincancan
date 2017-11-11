package org.iplatform.microservices.brain.service.multiLayernetwork;

import javax.annotation.PostConstruct;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.springframework.stereotype.Service;

@Service
public class Monitor {
    private UIServer uiServer;
    private StatsListener statsListener;

    public StatsListener getStatsListener() {
        return statsListener;
    }

    @PostConstruct
    public void init() {
        //初始化用户界面后端
        uiServer = UIServer.getInstance();
        //设置网络信息（随时间变化的梯度、分值等）的存储位置。这里将其存储于内存。
        StatsStorage statsStorage = new InMemoryStatsStorage(); //或者： new FileStatsStorage(File)，用于后续的保存和载入
        //将StatsStorage实例连接至用户界面，让StatsStorage的内容能够被可视化
        uiServer.attach(statsStorage);
        //然后添加StatsListener来在网络定型时收集这些信息
        statsListener = new StatsListener(statsStorage);
    }

}
