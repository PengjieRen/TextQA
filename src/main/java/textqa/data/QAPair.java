package textqa.data;

import nlp.Sentence;

public class QAPair {

	public Sentence query;
	public Sentence answer;
	public float label;
	public QAPair(Sentence query, Sentence anser, float label) {
		super();
		this.query = query;
		this.answer = anser;
		this.label = label;
	}
	
	
}
