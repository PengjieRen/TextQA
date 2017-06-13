package textqa.data;

import java.util.ArrayList;
import java.util.List;

import nlp.Sentence;
import nn4j.data.Data;

public class TextQAInstance extends Data{

	private List<QAPair> pairs;
	private String queryType;
	private Sentence query;
	
	public TextQAInstance(Sentence query, String queryType, List<Sentence> positiveAnswers,
			List<Sentence> negativeAnswers) {
		super();
		this.query = query;
		this.queryType = queryType;
		this.pairs=new ArrayList<QAPair>();
		for(Sentence a : positiveAnswers){
			pairs.add(new QAPair(query,a,1));
		}
		for(Sentence a : negativeAnswers){
			pairs.add(new QAPair(query,a,-1));
		}
	}
	public Sentence getQuery() {
		return query;
	}
	public String getQueryType() {
		return queryType;
	}
	public List<QAPair> getPairs() {
		return pairs;
	}
	public void setPairs(List<QAPair> pairs) {
		this.pairs = pairs;
	}
	
	
}
