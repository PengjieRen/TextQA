package textqa.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import nlp.INLPTools;
import nlp.Sentence;
import nlp.StanfordCoreNLPTool;
import nlp.Token;
import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.data.DataLoader;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;
import nn4j.expr.ParameterManager.Updater;
import nn4j.utils.StopHolder;
import nn4j.utils.VocabHolder;
import textqa.Constant;

public class WikiQALoader extends DataLoader {

	private VocabHolder vocab;
	private StopHolder stop;
	private List<TextQAInstance> data;

	private List<QAPair> pairs;
	private List<QAPair> ppairs;
	private List<QAPair> npairs;
	private Map<Sentence,List<QAPair>> query_ppairs;

	private int batchSize;
	private int pointer = 0;
	private int maxQuery;
	private int maxAnswer;

	public WikiQALoader(ParameterManager pm,File file, INLPTools nlp, VocabHolder vocab,StopHolder stop, int batchSize) {
		super(pm);
		this.batchSize = batchSize/2;
		this.stop=stop;
		this.vocab = vocab;
		data = load(file, nlp);
		pairs = new ArrayList<QAPair>();
		ppairs = new ArrayList<QAPair>();
		npairs = new ArrayList<QAPair>();
		query_ppairs=new HashMap<Sentence, List<QAPair>>();
		for (TextQAInstance ins : data) {
			for(QAPair pair:ins.getPairs()){
				if(pair.label>0){
					if(!query_ppairs.containsKey(pair.query)){
						query_ppairs.put(pair.query, new ArrayList<QAPair>());
					}
					query_ppairs.get(pair.query).add(pair);
					ppairs.add(pair);
				}else{
					npairs.add(pair);
				}
			}
			
			for(int i=0;i<ppairs.size();i++){
				pairs.add(ppairs.get(i));
			}
			
			for(int j=0;j<npairs.size();j++){
				pairs.add(npairs.get(j));
			}
		}
		for (QAPair pair : pairs) {
			int q = pair.query.getTokens().size();
			int a = pair.answer.getTokens().size();
			if (q > maxQuery) {
				maxQuery = q;
			}
			if (a > maxAnswer) {
				maxAnswer = a;
			}
		}
		
		shuffle(ppairs);
		shuffle(npairs);

	}
	
    protected void shuffle(List<QAPair> data){
		for(int i=0;i<data.size();i++){
			int index=rng.nextInt(data.size());
			QAPair p1=data.get(i);
			QAPair p2=data.get(index);
			data.set(index, p1);
			data.set(i, p2);
		}
	}
    
    public Batch toBatch(List<QAPair> pairs){
	    int thisBatchSize=pairs.size();
		
		Batch ret = new Batch();
		ret.batchInputs = new Parameter[2][];
		Parameter[] queryP = new Parameter[maxQuery];
		ret.batchInputs[0] = queryP;
		for (int i = 0; i < queryP.length; i++) {
			queryP[i] = new Parameter(Nd4j.zeros(thisBatchSize, vocab.dimension()), RegType.None, 0, false);
		}
		Parameter[] answerP = new Parameter[maxAnswer];
		ret.batchInputs[1] = answerP;
		for (int i = 0; i < answerP.length; i++) {
			answerP[i] = new Parameter(Nd4j.zeros(thisBatchSize, vocab.dimension()), RegType.None, 0, false);
		}
		ret.batchOtherInputs=new Parameter[1];
		ret.batchOtherInputs[0]=new Parameter(Nd4j.zeros(thisBatchSize, 1), RegType.None, 0, false);
		ret.batchMaskings = new INDArray[2];
		ret.batchMaskings[0]=Nd4j.zeros(thisBatchSize, maxQuery); 
		ret.batchMaskings[1]=Nd4j.zeros(thisBatchSize, maxAnswer); 
		ret.batchGroundtruth = Nd4j.zeros(thisBatchSize, 1);

		for (int i = 0; i < thisBatchSize;i++) {
			Set<String> qtokens=new HashSet<String>();
			List<Token> query = pairs.get(i).query.getTokens();
			for (int j = 0; j < Math.min(query.size(), maxQuery); j++) {
				((Parameter)ret.batchInputs[0][j]).value().putRow(i, vocab.toEmbed(query.get(j).getLower()));
				ret.batchMaskings[0].putScalar(i, j, 1);
				qtokens.add(query.get(j).getStem());
			}
			List<Token> answer = pairs.get(i).answer.getTokens();
			Set<String> atokens=new HashSet<String>();
			for (int j = 0; j < Math.min(answer.size(), maxAnswer); j++) {
				((Parameter)ret.batchInputs[1][j]).value().putRow(i, vocab.toEmbed(answer.get(j).getLower()));
				ret.batchMaskings[1].putScalar(i, j, 1);
				atokens.add(answer.get(j).getStem());
			}
			ret.batchGroundtruth.putScalar(i,0, pairs.get(i).label);
			
			int overlap=0;
			for(String t : qtokens){
				if(atokens.contains(t)){
					overlap++;
				}
			}
			((Parameter)ret.batchOtherInputs[0]).value().putScalar(i, 0,overlap);
		}
		
		return ret;
    }
	
	private Random rng=new Random();
	@Override
	public Batch next() {
		int thisBatchSize=pointer + batchSize <= npairs.size()? batchSize:(npairs.size()-pointer);
		List<QAPair> batchPairs=new ArrayList<QAPair>();
		for (int i = 0; i < thisBatchSize;i++) {
			QAPair pair=npairs.get(pointer+i);
			batchPairs.add(pair);
			
			List<QAPair> pairs=query_ppairs.get(pair.query);
			int rand=rng.nextInt(pairs.size());
			batchPairs.add(pairs.get(rand));
		}
		pointer+=thisBatchSize;

		return toBatch(batchPairs);
	}

	@Override
	public boolean hasNext() {
		return pointer < npairs.size();
	}

	@Override
	public void reset() {
		pointer = 0;
		shuffle(ppairs);
		shuffle(npairs);
	}

	@Override
	public List<Data> data() {
		List<Data> ret = new ArrayList<Data>();
		ret.addAll(data);
		return ret;
	}

	public List<TextQAInstance> load(File file, INLPTools nlp) {
		List<TextQAInstance> data = new ArrayList<TextQAInstance>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));

			Sentence query = null;
			String queryType = null;
			List<Sentence> positiveAnswers = new ArrayList<Sentence>();
			List<Sentence> negativeAnswers = new ArrayList<Sentence>();
			while (br.ready()) {
				String line = br.readLine();
				if (line.trim().length() == 0) {
					TextQAInstance ins = new TextQAInstance(query, queryType, positiveAnswers, negativeAnswers);
					data.add(ins);

					query = null;
					queryType = null;
					positiveAnswers = new ArrayList<Sentence>();
					negativeAnswers = new ArrayList<Sentence>();
					continue;
				}

				String[] temp = line.split("\t");
				if (temp.length == 4) {
					query = new Sentence(temp[1], nlp.tokenize(temp[1]), temp[3]);
					queryType = temp[2];
					query.id = temp[0];
				} else if (temp.length == 5) {
					Sentence answer = new Sentence(temp[3], nlp.tokenize(temp[3]), temp[4]);
					answer.id = temp[0];
					if (temp[1].equals("1")) {
						positiveAnswers.add(answer);
					} else {
						negativeAnswers.add(answer);
					}
				}
			}

			TextQAInstance ins = new TextQAInstance(query, queryType, positiveAnswers, negativeAnswers);
			data.add(ins);
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
		return data;
	}

	
	public static void main(String[] args) {
//		System.setErr(new PrintStream(new OutputStream() {
//			public void write(int b) {
//			}
//		}));
		
		VocabHolder vocab=new VocabHolder(new File(Constant.root,"model/glove.6B.50d.sample.txt"), null, null);
		StopHolder stop=new StopHolder(new File(Constant.root,"model/stop.dict"));
		INLPTools nlp=new StanfordCoreNLPTool("tokenize, ssplit, pos, lemma");
		ParameterManager pm = new ParameterManager(Updater.RMSPROP);

		WikiQALoader loader=new WikiQALoader(pm,new File(Constant.root,"data/Wiki-train.txt"), nlp, vocab,stop, 10);

		System.out.println(loader.maxQuery);
		System.out.println(loader.maxAnswer);
		int i=0;
		while(loader.hasNext()){
			Batch batch=loader.next();
			i++;
//			System.out.println(batch.batchInputs[0][0].value());
//			System.out.println(batch.batchInputs[1][0].value());
//			System.out.println(batch.batchMaskings[0]);
//			System.out.println(batch.batchMaskings[1]);
//			System.out.println(batch.batchGroundtruth);
//			System.out.println("----------------------------------------");
		}
		System.out.println(i);
	}
}
