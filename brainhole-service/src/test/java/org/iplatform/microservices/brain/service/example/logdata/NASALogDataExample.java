package org.iplatform.microservices.brain.service.example.logdata;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

import org.apache.commons.io.FilenameUtils;
//import org.apache.spark.SparkConf;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.api.java.JavaSparkContext;
//import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.LongColumnCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
//import org.datavec.spark.transform.AnalyzeSpark;
//import org.datavec.spark.transform.SparkTransformExecutor;
//import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

/**
 * 使用DataVec进行简单聚合操作预处理
 * 具体如下:
 * - 加载数据
 * - 数据质量分析
 * - 执行基本数据清洗
 * - 按主机记录进行分组，聚合计算请求数，请求字节 (such as number of requests and total number of bytes)
 * - 分析数据，打印结果
 *
 *
 * 数据来源: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
 *
 * 数据样例
 * 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
 * unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3985
 *
 * @author 张磊
 */
public class NASALogDataExample {
    public static final String DATA_PATH = "data/logdata/nasa/";//FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"datavec_log_example/");
    public static final String EXTRACTED_PATH = FilenameUtils.concat(DATA_PATH, "data");

    public static void main(String[] args) throws Exception {
//        downloadData();
//        SparkConf conf = new SparkConf();
//        conf.setMaster("local[*]");
//        conf.setAppName("DataVec Log Data Example");
//        JavaSparkContext sc = new JavaSparkContext(conf);
//
//        //定义数据模型
//        Schema schema = new Schema.Builder()
//                .addColumnString("host")
//                .addColumnString("timestamp")
//                .addColumnString("request")
//                .addColumnInteger("httpReplyCode")
//                .addColumnInteger("replyBytes")
//                .build();
//
//        //定义有效数据正则
//        String regex = "(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)";
//
//        //清除无效的数据
//        JavaRDD<String> logLines = sc.textFile(EXTRACTED_PATH);
//        logLines = logLines.filter(new Function<String, Boolean>() {
//            @Override
//            public Boolean call(String s) throws Exception {
//                //定义有效数据的正则
//                return s.matches(regex);
//            }
//        });
//
//        //解析原始数据并进行初步分析
//        RecordReader rr = new RegexLineRecordReader(regex, 0);
//        JavaRDD<List<Writable>> parsed = logLines.map(new StringToWritablesFunction(rr));
//
//        //检查数据质量
//        DataQualityAnalysis dqa = AnalyzeSpark.analyzeQuality(schema, parsed);
//        System.out.println("----- 数据质量检查 -----");
//        System.out.println(dqa); //此处会有一个数据问题，replyBytes出现非整数
//
//        //执行数据清理，解析，聚合
//        //使用datavec执行
//        Locale.setDefault(Locale.ENGLISH);
//        TransformProcess tp = new TransformProcess.Builder(schema)
//                //清理 replyBytes 这列的值，对于非整形的数据设置成0
//                .conditionalReplaceValueTransform("replyBytes", new IntWritable(0),
//                        new StringRegexColumnCondition("replyBytes", "\\D+"))
//                //解析日期时间字符串
//                .stringToTimeTransform("timestamp", "dd/MMM/YYYY:HH:mm:ss Z", DateTimeZone.forOffsetHours(-4))
//                //按主机进行reduce进行汇聚
//                .reduce(new Reducer.Builder(ReduceOp.CountUnique).keyColumns("host") //按主机列进行分组
//                        .countColumns("timestamp") //计数
//                        .countUniqueColumns("request", "httpReplyCode") //request 和 sreply codes 联合主键
//                        .sumColumns("replyBytes") //对replyBytes求和
//                        .build())
//                .renameColumn("count", "numRequests")
//                //过滤掉请求小于100完字节的请求
//                .filter(new ConditionFilter(new LongColumnCondition("sum(replyBytes)", ConditionOp.LessThan, 1000000)))
//                .build();
//
//        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsed, tp);
//        processed.cache();
//
//        //分析结果
//        Schema finalDataSchema = tp.getFinalSchema();
//        long finalDataCount = processed.count();
//        List<List<Writable>> sample = processed.take(10);
//        DataAnalysis analysis = AnalyzeSpark.analyze(finalDataSchema, processed);
//        System.out.println("----- Final Data Schema -----");
//        System.out.println(finalDataSchema);
//
//        System.out.println("\n\nFinal data count: " + finalDataCount);
//
//        System.out.println("\n\n----- Samples of final data -----");
//        for (List<Writable> l : sample) {
//            System.out.println(l);
//        }
//        System.out.println("\n\n----- Analysis -----");
//        System.out.println(analysis);
//
//        Scanner input = new Scanner(System.in);
//        String val = null;
//        do{
//            System.out.print("请输入(退出输入exit)：");
//            val = input.next();
//        }while(!val.equals("exit"));
//        input.close();
//        sc.stop();
    }

    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists())
            directory.mkdir();

        //Download file:
        String archivePath = DATA_PATH + "NASA_access_log_Jul95.gz";
        File archiveFile = new File(archivePath);
        File extractedFile = new File(EXTRACTED_PATH, "access_log_July95.txt");
        new File(extractedFile.getParent()).mkdirs();

        //Assume if archive (.tar.gz) exists, then data has already been extracted
        System.out.println("Data (.gz file) already exists at " + archiveFile.getAbsolutePath());
        if (!extractedFile.exists()) {
            //Extract tar.gz file to output directory
            extractGzip(archivePath, extractedFile.getAbsolutePath());
        } else {
            System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
        }
    }

    private static final int BUFFER_SIZE = 4096;

    private static void extractGzip(String filePath, String outputPath) throws IOException {
        System.out.println("Extracting files...");
        byte[] buffer = new byte[BUFFER_SIZE];

        try {
            GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(new File(filePath)));

            FileOutputStream out = new FileOutputStream(new File(outputPath));

            int len;
            while ((len = gzis.read(buffer)) > 0) {
                out.write(buffer, 0, len);
            }

            gzis.close();
            out.close();

            System.out.println("Done");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

}