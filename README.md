# logistic-regression
A Simple Logistic Regression In Java.

## Test
```java
public static void main(String[] args) throws Exception {
	
	LogisticRegression lr = new LogisticRegression(4, 0.01);
	
	double[][] xs = new double[][] {
					{5.1, 3.5, 1.4, 0.2},
					{4.7, 3.2, 1.3, 0.2},
					{4.6, 3.1, 1.5, 0.2},
					{5.4, 3.4, 1.7, 0.2},
					{4.9, 3.1, 1.5, 0.1},
					{5.0, 2.0, 3.5, 1.0},
					{6.1, 2.9, 4.7, 1.4},
					{6.1, 2.8, 4.7, 1.2},
					{5.7, 2.8, 4.1, 1.3},
					{6.5, 3.0, 5.5, 1.8}
	};
	int[] ys = new int[] {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
	lr.train(xs, ys);

	// w1: 0.6787525194211941 w2: 1.9503532862744557 w3: -2.9734436461122473 w4: -1.2954640647320277
	System.out.println("w1: " + lr.ws[0] + " w2: " +lr.ws[1] + " w3: " + lr.ws[2] + " w4: " + lr.ws[3]);
	System.out.println(lr.predict(new double[] {4.6,3.2,1.4,0.2})); // 0.9929066448006123
	System.out.println(lr.predict(new double[] {5.9,3.0,5.1,1.8})); // 4.8026209792515905E-4
}
```
