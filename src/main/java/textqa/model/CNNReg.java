package textqa.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nlp.INLPTools;
import nlp.StanfordCoreNLPTool;
import nn4j.cg.CNN;
import nn4j.cg.Dense;
import nn4j.cg.PoolingType;
import nn4j.data.Batch;
import nn4j.expr.Concat;
import nn4j.expr.DefaultParamInitializer;
import nn4j.expr.Expr;
import nn4j.expr.InnerProduct;
import nn4j.expr.ParamInitializer;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;
import nn4j.expr.ParameterManager.Updater;
import nn4j.expr.WeightInit;
import nn4j.loss.Loss;
import nn4j.utils.StopHolder;
import nn4j.utils.VocabHolder;
import textqa.Constant;
import textqa.data.QAPair;
import textqa.data.WikiQALoader;

public class CNNReg extends RegModel{

	private Parameter w1;
	private Parameter w2;
	private Parameter w3;
	private Parameter w;

	private WikiQALoader loader;
	public CNNReg(ParameterManager pm,WikiQALoader loader) {
		super(pm);
		this.loader=loader;
	}
	
	@Override
	public void parameters() {
		w = pm.createParameter(new DefaultParamInitializer(WeightInit.DISTRIBUTION,new UniformDistribution(-0.01, 0.01)).init(new int[] { 100, 50 }),RegType.None,0, true);
		w1 = pm.createParameter(new DefaultParamInitializer(WeightInit.DISTRIBUTION,new UniformDistribution(-0.3, 0.3)).init(new int[] { 3, 1 }),RegType.None,0, true);
//		w2 = pm.createParameter(init.init(new int[] { 101, 100 }),RegType.None,0, true);
//		w3 = pm.createParameter(init.init(new int[] { 101, 1 }),RegType.None,0, true);
	}
	
	@Override
	public Loss model(Batch batch, boolean training) {
		CNN qcnn=new CNN(batch.batchMaskings[0],batch.batchInputs[0],w,2,PoolingType.Avg,training);
		CNN acnn=new CNN(batch.batchMaskings[1],batch.batchInputs[1],w,2,PoolingType.Avg,training);
		
		Expr dot=new InnerProduct(qcnn, acnn);
		
		Expr v1 = new Dense(new Concat(dot,batch.batchOtherInputs[0]),w1,Activation.TANH,true,training);
//		Expr v2 = new Dense(v1,w2,Activation.TANH,true,training);
//		Expr v3 = new Dense(v2,w3,Activation.TANH,true,training);
		Loss loss = new Loss(v1, LossFunction.HINGE);
		return loss;
	}

	@Override
	public float score(QAPair pair) {
		List<QAPair> pairs=new ArrayList<QAPair>();
		pairs.add(pair);
		Loss loss=model(loader.toBatch(pairs),false);
		return loss.forward().getFloat(0);
	}

	public static void main(String[] args) {
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
		System.setProperty("ndarray.order","c");

//		CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).allowCrossDeviceAccess(false).useDevices(0,1);

//		System.setErr(new PrintStream(new OutputStream() {
//			public void write(int b) {
//			}
//		}));
		
		VocabHolder vocab=new VocabHolder(new File(Constant.root,"model/glove.6B.50d.sample.txt"), null, null);
		StopHolder stop=new StopHolder(new File(Constant.root,"model/stop.dict"));
		INLPTools nlp=new StanfordCoreNLPTool("tokenize, ssplit, pos, lemma");
		ParameterManager pm = new ParameterManager(Updater.RMSPROP);

		WikiQALoader trainLoader=new WikiQALoader(pm,new File(Constant.root,"data/Wiki-train.txt"), nlp, vocab,stop, 20);
		WikiQALoader devLoader=new WikiQALoader(pm,new File(Constant.root,"data/Wiki-dev.txt"), nlp, vocab,stop, 20);
		WikiQALoader testLoader=new WikiQALoader(pm,new File(Constant.root,"data/Wiki-test.txt"), nlp, vocab,stop, 20);

		CNNReg model=new CNNReg(pm,testLoader);
		model.train(trainLoader, devLoader.data(), new File(Constant.root,"data/Wiki-dev.gt.txt"), testLoader.data(), new File(Constant.root,"data/Wiki-test.gt.txt"), 100);
	}

}
