package org.iplatform.microservices.brain.service.example.logdata;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeZone;

/**
 * @author 张磊
 */
public class VMWareLogDataExample {
    public static final String DATA_PATH = "data/logdata/vmware/CLDS_VC_EVENT-UTF8.csv";
    //public static final String DATA_PATH = "data/logdata/vmware/split_aa.csv";

    public static void main(String[] args) throws Exception {
//        try {
//            //in
//            FileInputStream in = new FileInputStream("data/logdata/vmware/CLDS_VC_EVENT.csv");
//            InputStreamReader inReader = new InputStreamReader(in, "GB2312");
//            BufferedReader bufReader = new BufferedReader(inReader);
//            //out
//            FileOutputStream out = new FileOutputStream(DATA_PATH);
//            OutputStreamWriter outWriter = new OutputStreamWriter(out, "UTF-8");
//            BufferedWriter bufWrite = new BufferedWriter(outWriter);
//
//            String line = null;
//            String mline = null;
//            while ((line = bufReader.readLine()) != null) {
//                if (line.endsWith("\"")) {
//                    if (mline != null) {
//                        bufWrite.write(mline + line + "\r\n");
//                        mline = null;
//                    } else {
//                        bufWrite.write(line + "\r\n");
//                    }
//                } else {
//                    if (mline == null) {
//                        mline = line;
//                    } else {
//                        mline = mline + line;
//                    }
//                }
//            }
//            //in
//            bufReader.close();
//            inReader.close();
//            in.close();
//            //out
//            bufWrite.close();
//            outWriter.close();
//            out.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Log Data Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //定义数据模型
        Schema schema = new Schema.Builder().addColumnString("EVENT_ID").addColumnString("EVENT_TIME")
                .addColumnString("INDB_TIME").addColumnString("EVENT_DESC").addColumnString("EVENT_CLASS")
                .addColumnString("VC_ID").addColumnsString("VC_DATACENTER_ID", "VC_CLUSTER_ID", "VC_HOST_ID",
                        "VC_DATASTORE_ID", "VC_USER", "VC_VM_ID", "VC_NET_ID", "VC_DVS_ID", "OBJECT_ID")
                .build();

        //清除无效的数据
        JavaRDD<String> logLines = sc.textFile(DATA_PATH);
        logLines = logLines.map(new Function<String, String>() {
            String datetime_regex = "\\d{4}/\\d{1,2}/\\d{1,2} \\d{1,2}:\\d{1,2}:\\d{1,2}";
            String date_regex = "\\d{4}/\\d{1,2}/\\d{1,2}";

            @Override
            public String call(String rawline) throws Exception {
                //System.out.println(rawline);
                String[] rawlineFields = rawline.split(",");
                String EVENT_TIME = rawlineFields[1];
                EVENT_TIME = EVENT_TIME.substring(1, EVENT_TIME.length() - 1);
                if (EVENT_TIME.matches(datetime_regex)) {
                    return rawline;
                } else if (EVENT_TIME.matches(date_regex)) {
                    rawlineFields[1] = "\"" + EVENT_TIME + " 00:00:00\"";
                    return String.join(",", rawlineFields);
                } else {
                    return rawline;
                }
            }

        });
        //        logLines = logLines.filter(new Function<String, Boolean>() {
        //            String regex = "\\d{4}/\\d{1,2}/\\d{1,2} \\d{1,2}:\\d{1,2}:\\d{1,2}";
        //            @Override
        //            public Boolean call(String s) throws Exception {
        //                //定义有效数据的正则
        //                String line = new String(s.getBytes("GB2312"));
        //                String EVENT_TIME = s.split(",")[1];
        //                EVENT_TIME = EVENT_TIME.substring(1, EVENT_TIME.length()-1);
        //                return EVENT_TIME.matches(regex); 
        //            }
        //        });

        //解析原始数据并进行初步分析
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsed = logLines.map(new StringToWritablesFunction(rr));

        //检查数据质量
        DataQualityAnalysis dqa = AnalyzeSpark.analyzeQuality(schema, parsed);
        System.out.println("----- 数据质量检查 -----");
        System.out.println(dqa);

        //执行数据清理，解析，聚合
        //使用datavec执行
        Locale.setDefault(Locale.ENGLISH);
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("EVENT_ID", "INDB_TIME", "VC_DATACENTER_ID", "VC_CLUSTER_ID", "VC_HOST_ID",
                        "VC_DATASTORE_ID", "VC_USER", "VC_VM_ID", "VC_NET_ID", "VC_DVS_ID", "OBJECT_ID")
                .stringToTimeTransform("EVENT_TIME", "YYYY/MM/DD HH:mm:ss", DateTimeZone.forOffsetHours(0))
                .reduce(new Reducer.Builder(ReduceOp.CountUnique).keyColumns("VC_ID").countColumns("EVENT_TIME")
                        .countUniqueColumns("EVENT_CLASS", "EVENT_DESC").build())
                .build();

        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsed, tp);
        JavaRDD<String> processedAsString = processed.map(new WritablesToStringFunction(","));
        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = logLines.collect();
        //processed.cache();

        //分析结果
        Schema finalDataSchema = tp.getFinalSchema();
        long finalDataCount = processed.count();
        List<List<Writable>> sample = processed.take(10);
        DataAnalysis analysis = AnalyzeSpark.analyze(finalDataSchema, processed);
        System.out.println("----- Final Data Schema -----");
        System.out.println(finalDataSchema);

        System.out.println("\n\nFinal data count: " + finalDataCount);

        System.out.println("\n\n----- Samples of final data -----");
        for (List<Writable> l : sample) {
            System.out.println(l);
        }
        System.out.println("\n\n----- Analysis -----");
        System.out.println(analysis);

        Scanner input = new Scanner(System.in);
        String val = null;
        do {
            System.out.print("请输入(退出输入exit)：");
            val = input.next();
        } while (!val.equals("exit"));
        input.close();
        sc.stop();
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