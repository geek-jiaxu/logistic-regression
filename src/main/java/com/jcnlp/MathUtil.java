package com.jcnlp;

public class MathUtil {

	public static double sigmoid(double y) {
		return 1.0 / (1.0 + Math.exp(- y));
	}
}
