package textqa.model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import nn4j.cg.ComputationGraph;
import nn4j.data.Data;
import nn4j.expr.ParameterManager;
import textqa.Constant;
import textqa.data.QAPair;
import textqa.data.TextQAInstance;
import treceval.TrecEval;

public abstract class RegModel extends ComputationGraph {

	public RegModel(ParameterManager pm) {
		super(pm);
	}

	protected int numHid = 50;
	protected TrecEval eval = new TrecEval();
	protected float acceptProb = 0.8f;
	protected boolean training;

	@Override
	public void test(String run, List<Data> test, File gtFile) {
		training = false;
		try {
			File outFile = new File(Constant.root, "eval/" + run);
			BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));
			for (Data data : test) {
				TextQAInstance ins = (TextQAInstance) data;
				List<QAPair> pairs = ins.getPairs();

				Map<QAPair, Float> results = new HashMap<QAPair, Float>();
				for (int i = 0; i < pairs.size(); i++) {
					results.put(pairs.get(i), score(pairs.get(i)));
				}

				List<Map.Entry<QAPair, Float>> list = new ArrayList<Map.Entry<QAPair, Float>>(results.entrySet());
				Collections.sort(list, new Comparator<Map.Entry<QAPair, Float>>() {
					public int compare(Entry<QAPair, Float> o1, Entry<QAPair, Float> o2) {
						return o2.getValue().compareTo(o1.getValue());
					}
				});

				int rank = 0;
				for (Map.Entry<QAPair, Float> entry : list) {
					bw.write(entry.getKey().query.id + " Q0 " + entry.getKey().answer.id + " " + rank + " "
							+ entry.getValue() + " " + this.getClass().getName());
					bw.newLine();
					rank++;
				}
			}
			bw.close();

			// eval
			eval.eval(new String[] { "-c", "-m", "recip_rank", "-m", "map", gtFile.getAbsolutePath(),
					outFile.getAbsolutePath() });
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public abstract float score(QAPair pair);
}
