# Intro to Machine Learning

Video Timelines for the 12 lessons of “Intro to Machine Learning” by EricPB

https://forums.fast.ai/t/another-treat-early-access-to-intro-to-machine-learning-videos/6826/319

Lesson 1 video timeline
00:02:14 83 AWS or Crestle Deep Learning


00:05:14 18 lesson1-rf notebook Random Forests


00:10:14 2 ?display documentation, ??display source code


00:12:14 11 Blue Book for Bulldozers Kaggle competition: predict auction sale price,
 Download Kaggle data to AWS using a nice trick with FireFox javascript console, getting a full cURL link,
 Using Jupyter “New Terminal”


00:23:55 4 using !ls {PATH} in Jupyter Notebook


00:26:14 7 Structured Data vs data like Computer Vision, NLP, Audio,
 ‘vimimports.py′in/fastai, ‘low_memory=False’, ‘parse_dates’,
 Python 3.6 format string f’{PATH}Train.csv’,
 ‘display_all()’


00:33:14 13 Why Jeremy’s doesn’t do a lot of EDA,
 Bulldozer RMSLE difference between the log of prices


00:36:14 9 Intro to Random Forests, in general doesn’t overfit, no need to setup a validation set.
 The Silly Concepts of Cursive Dimensionality and No Free Lunch theorem,
 Brief history of ML and lots of theory vs practice in the 90’s.


00:43:14 9 RandomForestRegressor, RandomForestClassifier
 Stack Trace: how to fix an error


00:48:14 3 Continuous and categorical variables, add_datepart()


00:57:14 5 Dealing with strings in data (“low, medium, high” etc.), which must be converted into numeric coding, with train_cats() creating a mapping of integers to the strings.
 Warning: make sure to use the same mapping string-numbers in Training and Test sets,
 Use “apply_cats” for that,
 Change order of index of .cat.categories with .cat.set_categories.


01:07:14 5 Pre-processing to replace categories with their numeric codes,
 Handle missing continuous values,
 And split the dependant variable into a separate variable.
 proc_df() and fix_missing()


01:14:01 4 ‘split_vals()’




Lesson 2
00:03:30 31 simlink sim link to fastai directory


00:06:15 12 understand the RMSLE relation to RMSE, and why use np.log(‘SalePrice’) with RMSE as a result


00:09:01 9 proc_df, numericalize


00:11:01 2 rsquare root square of mean errors RMSE,
 What the formula rsquare (and others in general) does and understand it


00:17:30 6 Creating a good validation set, ‘split_vals()’ explained
 “I don’t trust ML, we tried it, it looked great, we put it in production, it didn’t work” because the validation set was not representative !


00:21:01 4 overfitting over-fitting underfitting ‘don’t look at test set !’,
 Example of failed methodology in sociology, psychology,
 Hyperparameters,
 Using PEP8 (or not) for ML prototyping models


00:29:01 7 RMSE function and RandomForestRegressor,
 Speeding things up with a smaller dataset (subset = ),
 Use of ‘_’ underscore in Python


00:32:01 4 Single Tree model and visualize it,
 max_depth=3,
 bootstrap=False


00:47:01 7 Bagging of little Boostraps, ensembling


00:57:01 1 scikit-learn ExtraTreeRegressor randomly tries variables


01:04:01 3 m.estimators_,
 Using list comprehension


01:10:00 7 Out-of-bag (OOB) score


01:13:45 9 Automate hyperparameters hyper-parameters with grid-search gridsearch
 Randomly subsample the dataset to reduce overfitting with ‘set_rf_samples()’, code detail at 1h18m25s


01:17:20 2 Tip for Favorita Grocery competition,
 ‘set_rf_samples()’,
 ‘reset_rf_samples()’,
 ‘min_samples_leaf=’,
 ‘max_features=’


01:30:20 2 Looking at ‘fiProductClassDesc’ column with .cat.categories and .cat.codes




Lesson 3
00:02:44 21 When to use or not Random Forests (unstructured data like CV or Sound works better with DL),
 Collaborative filtering for Favorita


00:05:10 8 dealing with missing values present in Test but not Train (or vice-versa) in ‘proc_df()’ with “nas” dictionary whose keys are names of columns with missing values, and the values are the medians.


00:09:30 2 Starting Favorita notebook,
 The ability to explain the goal of a Kaggle competition or a project,
 What are independent and dependant variables ?
 Star schema warehouse database, snowflake schema


00:15:30 3 Use dtypes to read data without ‘low_memory = False’


00:20:30 7 Use ‘shuf’ to read a sample of large dataset at start


00:26:30 3 Take the Log of the sales with ‘np.log1p()’,
 Apply ‘add_datepart)’,
 ‘split_vals(a,n)’,


00:28:30 5 Models,
 ‘set_rf_samples’,
 ‘np.array(trn, dtype=np.float32’,
 Use ‘%prun’ to find lines of code that takes a long time to run


00:33:30 4 We only get reasonable results, but nothing great on the leaderboard: WHY ?


00:43:30 4 Quick look at Rossmann grocery competition winners,
 Looking at the choice of validation set with Favorita Leaderboard by Terence Parr (his @ pseudo here ?)


00:50:30 9 Lesson2-rf interpretation,
 Why is ‘nas’ an input AND an output variable in ‘proc_df()’


00:55:30 13 How confident are we in our predictions (based on tree variance) ?
 Using ‘set_rf_samples()’ again.
 ‘parallel_trees()’ for multithreads parallel processing,
 EROPS, OROPS, Enclosure


01:07:15 5 Feature importance with ‘rf_feat_importance()’


01:12:15 5 Data leakage example,
 Colinearity




Lesson 4
00:00:04 18 How to deal with version control and notebooks ? Make a copy and rename it with “tmp-blablabla” so it’s hidden from Git Pull


00:01:50 11 Summarize the relationship between hyperparameters in Random Forests, overfitting and colinearity.
 ‘set_rf_samples()’, ‘oob_score = True’,
 ‘min_samples_leaf=’ 8m45s,
 ‘max_features=’ 12m15s


00:18:50 6 Random Forest Interpretation lesson2-rf_interpretation,
 ‘rf_feat_importance()’


00:26:50 3 ‘to_keep = fi[fi.imp>0.005]’ to remove less important features,
 high cardinality variables 29m45s,


00:32:15 6 Two reasons why Validation Score is not good or getting worse: overfitting, and validation set is not a random sample (something peculiar in it, not in Train),
 The meaning of the five numbers results in ‘print_score(m)’, RMSE of Training & Validation, R² of Train & Valid & OOB.
 We care about the RMSE of Validation set.


00:35:50 5 How Feature Importance is normally done in Industry and Academics outside ML: they use Logistic Regression Coefficients, not Random Forests Feature/Variable Importance.


00:39:50 3 Doing One-hot encoding for categorical variables,
 Why and how works ‘max_n_cat=7’ based on Cardinality 49m15s, ‘numericalize’


00:55:05 4 Removing redundant features using a dendogram and '.spearmanr()'for rank correlation, ‘get_oob(df)’, ‘to_drop = []’ variables, ‘reset_rf_samples()’


01:07:15 4 Partial dependence: how important features relate to the dependent variable, ‘ggplot() + stat_smooth()’, ‘plot_pdp()’


01:21:50 What is the purpose of interpretation, what to do with that information ?


01:30:15 5 What is EROPS / OROPS ?


01:32:25 5 Tree interpreter




Lesson 5
00:00:04 18 Review of Training, Test set and OOB score, intro to Cross-Validation (CV),
 In Machine Learning, we care about Generalization Accuracy/Error.


00:11:35 3 Kaggle Public and Private test sets for Leaderboard,
 the risk of using a totally random validation set, rerun the model including Validation set.


00:22:15 3 Is my Validation set truly representative of my Test set. Build 5 very different models and score them on Validation and on Test. Examples with Favorita Grocery.


00:28:10 Why building a representative Test set is crucial in the Real World machine learning (not in Kaggle),
 Sklearn make train/test split or cross-validation = bad in real life (for Time Series) !


00:31:04 6 What is Cross-Validation and why you shouldn’t use it most of the time (hint: random is bad)


00:38:04 8 Tree interpretation revisited, lesson2-rf_interpreter.ipynb, waterfall plot for increase and decrease in tree splits,
 ‘ti.predict(m, row)’


00:48:50 10 Dealing with Extrapolation in Random Forests,
 RF can’t extrapolate like Linear Model, avoid Time variables as predictors if possible ?
 Trick: find the differences between Train and Valid sets, ie. any temporal predictor ? Build a RF to identify components present in Valid only and not in Train ‘x,y = proc_df(df_ext, ‘is_valid’)’,
 Use it in Kaggle by putting Train and Test sets together and add a column ‘is_test’, to check if Test is a random sample or not.


00:59:15 2 Our final model of Random Forests, almost as good as Kaggle #1 (Leustagos & Giba)


01:03:04 What to expect for the in-class exam


01:05:04 Lesson3-rf_foundations.ipynb, writing our own Random Forests code.
 Basic data structures code, class ‘TreeEnsemble()’, np.random.seed(42)’ as pseudo random number generator
 How to make a prediction in Random Forests (theory) ?


01:21:04 7 class ‘DecisionTree()’,
 Bonus: Object-Oriented-Programming (OOP) overview, critical for PyTorch




Lesson 6
Note: this lesson has a VERY practical discussion with USF students about the use of Machine Learning in business/corporation, Jeremy shares his experience as a business consultant (McKinsey) and entrepreneur in AI/ML. Deffo not PhD’s stuff, too real-life.
00:00:04 20 Review of previous lessons: Random Forests interpretation techniques,
 Confidence based on tree variance,
 Feature importance,
 Removing redundant features,
 Partial dependence…
 And why do we do Machine Learning, what’s the point ?
 Looking at PowerPoint ‘intro.ppx’ in Fastai GitHub: ML applications (horizontal & vertical) in real-life.
 Churn (which customer is going to leave) in Telecom: google “jeremy howard data products”,
 drive-train approach with ‘Defined Objective’ -> ‘Company Levers’ -> ‘Company Data’ -> ‘Models’


00:10:01 3 "In practice, you’ll care more about the results of your simulation than your predictive model directly ",
 Example with Amazon 'not-that-smart’recommendations vs optimization model.
 More on Churn and Machine Learning Applications in Business


00:20:30 Why is it hard/key to define the problem to solve,
 ICYMI: read “Designing great data products” from Jeremy in March 28, 2012 ^!^
 Healthcare applications like ‘Readmission risk’. Retail applications examples.
 There’s a lot more than what you read about Facebook or Google applications in Tech media.
 Machine Learning in Social Sciences today: not much.


00:37:15 More on Random Forests interpretation techniques.
 Confidence based on tree variance


00:42:30 4 Feature importance, and Removing redundant features


00:50:45 Partial dependence (or dependance)


01:02:45 4 Tree interpreter (and a great example of effective technical communications by a student)
 Using Excel waterfall chart from Chris
 Using ‘hub.github.com 6’, a command-line wrapper for git that makes you better at GitHub.


01:16:15 5 Extrapolation, with a 20 mins session of live coding by Jeremy




Lesson 7
00:00:01 18 Review of Random Forest previous lessons,
 Lots of historical/theoritical techniques in ML that we don’t use anymore (like SVM)
 Use of ML in Industry vs Academia, Decision-Trees Ensemble


00:05:30 2 How big the Validation Set needs to be ? How much the accuracy of your model matters ?
 Demo with Excel, T-distribution and n>22 observations in every class
 Standard Deviation : np(1-p), Standard Error (stdev mean): stdev/sqrt(n)


00:18:45 2 Back to Random Forest from scratch.
 “Basic data structures” reviewed


00:32:45 1 Single Branch
 Find the best split given variable with ‘find_better_split’, using Excel demo again


00:45:30 Speeding things up


00:55:00 Full single tree


01:01:30 Predictions with ‘predict(self,x)’,
 and ‘predict_row(self, xi)’


01:09:05 Putting it all together,
 Cython an optimising static compiler for Python and C


01:18:01 3 “Your mission, for next class, is to implement”:
 Confidence based on tree variance,
 Feature importance,
 Partial dependence,
 Tree interpreter.


01:20:15 Reminder: How to ask for Help on Fastai forums
 http://wiki.fast.ai/index.php/How_to_ask_for_Help 1
 Getting a screenshot, resizing it.
 For lines of code, create a “Gist”, using the extension ‘Gist-it’ for “Create/Edit Gist of Notebook” with ‘nbextensions_configurator’ on Jupyter Notebook, ‘Collapsible Headings’, ‘Chrome Clipboard’, ‘Hide Header’


01:23:15 9 We’re done with Random Forests, now we move on to Neural Networks.
 Random Forests can’t extrapolate, it just averages data that it has already seen, Linear Regression can but only in very limited ways.
 Neural Networks give us the best of both worlds.
 Intro to SGD for MNIST, unstructured data.
 Quick comparison with Fastai/Jeremy’s Deep Learning Course.




Lesson 8
00:00:45 22 Moving from Decision Trees Ensemble to Neural Nets with Mnist
 lesson4-mnist_sgd.ipynb notebook


00:08:20 3 About Python ‘pickle()’ pros & cons for Pandas, vs ‘feather()’,
 Flatten a tensor


00:13:45 1 Reminder on the jargon: a vector in math is a 1d array in CS,
 a rank 1 tensor in deep learning.
 A matrix is a 2d array or a rank 2 tensor, rows are axis 0 and columns are axis 1


00:17:45 2 Normalizing the data: subtracting off the mean and dividing by stddev
 Important: use the mean and stddev of Training data for the Validation data as well.
 Use the ‘np.reshape()’ function


00:34:25 2 Slicing into a tensor, ‘plots()’ from Fastai lib.


00:38:20 2 Overview of a Neural Network
 Michael Nielsen universal approximation theorem: a visual proof that neural nets can compute any function
 Why you should blog (by Rachel Thomas)


00:47:15 1 Intro to PyTorch & Nvidia GPUs for Deep Learning
 Website to buy a laptop with a good GPU: xoticpc.com 4
 Using cloud services like Crestle.com 1 or AWS (and how to gain access EC2 w/ “Request limit increase”)


00:57:45 7 Create a Neural Net for Logistic Regression in PyTorch
 ‘net = nn.Sequential(nn.Linear(28*28, 10), nn.LogSoftmax()).cuda()’
 ‘md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid))’
 Loss function such as ‘nn.NLLLoss()’ or Negative Log Likelihood Loss or Cross-Entropy (binary or categorical)
 Looking at Loss with Excel


01:09:05 Let’s fit the model then make predictions on Validation set.
 ‘fit(net, md, epochs=1, crit=loss, opt=opt, metrics=metrics)’
 Note: PyTorch doesn’t use the word “loss” but the word “criterion”, thus ‘crit=loss’
 ‘preds = predict(net, md.val_dl)’
 ‘preds.shape’ -> (10000, 10)
 ‘preds.argmax(axis=1)[:5]’, argmax will return the index of the value which is the number itself.
 ‘np.mean(preds == y_valid)’ to check how accurate the model is on Validation set.


01:16:05 3 A second pass on “Michael Nielsen universal approximation theorem”
 A Neural Network can approximate any other function to close accuracy, as long as it’s large enough.


01:18:15 1 Defining Logistic Regression ourselves, from scratch, not using PyTorch ‘nn.Sequential()’
 Demo explanation with drawings by Jeremy.
 Look at Excel ‘entropy_example.xlsx’ for Softmax and Sigmoid


01:31:05 3 Assignements for the week, student question on ‘Forward(self, x)’




Lesson 9
Jeremy starts with a selection of students’ posts.
00:00:01 18 Structuring the Unstructured: a visual demo of Bagging with Random Forests.
 http://structuringtheunstructured.blogspot.se/2017/11/coloring-with-random-forests.html 8


00:04:01 3 Parfit: a library for quick and powerful hyper-parameter optimization with visualizations.
 . How to make SGD Classifier perfomr as well as Logistic Regression using Parfit
 . Intuitive Interpretation of Random Forest
 . Statoil/C-Core Iceberg Classifier Challenge on Kaggle: a Keras Model for Beginners + EDA


Back to the course.
00:09:01 2 Why write a post on your learning experience, for you and for newcomers.


00:09:50 7 Using SGD on MNIST for digit recognition
 . lesson4-mnist_sgd.ipynb notebook


00:11:30 1 Training the simplest Neural Network in PyTorch
 (long step-by-step demo, 30 mins approx)


00:46:55 4 Intro to Broadcasting: “The MOST important programming concept in this course and in Machine Learning”
 . Performance comparison between C and Python
 . SIMD: “Single Instruction Multiple Data”
 . Multiple processors/cores and CUDA


00:52:10 Broadcasting in details


01:05:50 Broadcasting goes back to the days of APL (1950’s) and Jsoftware
 . More on Broadcasting


01:12:30 Matrix Multiplication -and not-.
 . Writing our own training loop.


Lesson 10
00:00:01 13 Fast.ai is now available on PIP !
 And more USF students publications: class-wise Processing in NLP, Class-wise Regex Functions
 . Porto Seguro’s Safe Driver Prediction (Kaggle): 1st place solution with zero feature engineering !
 Dealing with semi-supervised-learning (ie. labeled and unlabeled data)
 Data augmentation to create new data examples by creating slightly different versions of data you already have.
 In this case, he used Data Augmentation by creating new rows with 15% randomly selected data.
 Also used “auto-encoder”: the independant variable is the same as the dependant variable, as in “try to predict your input” !


00:08:30 2 Back to a simple Logistic Regression with MNIST summary
 ‘lesson4-mnist_sgd.ipynb’ notebook


00:11:30 PyTorch tutorial on Autograd


00:15:30 1 “Stream Processing” and “Generator Python”
 . “l.backward()”
 . “net2 = LogReg().cuda()”


00:32:30 3 Building a complete Neural Net, from scratch, for Logistic Regression in PyTorch, with “nn.Sequential()”


00:58:00 1 Fitting the model in ‘lesson4-mnist_sgd.ipynb’ notebook
 The secret in modern ML (as covered in the Deep Learning course): massively over-paramaterized the solution to your problem, then use Regularization.


01:02:10 1 Starting NLP with IMDB dataset and the sentiment classification task
 NLP = Natural Language Processing


01:03:10 2 Tokenizing and ‘term-document matrix’ & "Bag-of-Words’ creation
 “trn, trn_y = texts_from_folders(f’{PATH}train’, names)” from Fastai library to build arrays of reviews and labels
 Throwing the order of words with Bag-of-Words !


01:08:50 1 sklearn “CountVectorizer()”
 “fit_transform(trn)” to find the vocabulary in the training set and build a term-document matrix.
 “transform(val)” to apply the same transformation to the validation set.


01:12:30 What is a ‘sparse matrix’ to store only key info and save memory.
 More details in Rachel’s “Computational Algebra” course on Fastai


01:16:40 2 Using “Naive Bayes” for “Bag-of-Words” approaches.
 Transforming words into features, and dealing with the bias/risk of “zero probabilities” from the data.
 Some demo/discussion about calculating the probabilities of classes.


01:25:00 1 Why is it called “Naive Bayes”


01:30:00 The difference between theory and practice for “Naive Bayes”
 Using Logistic regression where the features are the unigrams


01:35:40 Using Bigram & Trigram with Naive Bayes (NB) features


Lesson 11
00:00:01 14 Review of optimizing multi-layer functions with SGD
 “d(h(g(f(x)))) / dw = 0,6”


00:09:45 3 Review of Naive Bayes & Logistic Regression for NLP with lesson5-nlp.ipynb notebook


00:16:30 1 Cross-Entropy as a popular Loss Function for Classification (vs RMSE for Regression)


00:21:30 1 Creating more NLP features with Ngrams (bigrams, trigrams)


00:23:01 5 Going back to Naive Bayes and Logistic Regression,
 then ‘We do something weird but actually not that weird’ with “x_nb = x.multiply®”
 Note: watch the whole 15 mins segment for full understanding.


00:39:45 3 ‘Baselines and Bigrams: Simple, Good Sentiment and Topic Classification’ paper by Sida Wang and Christopher Manning, Stanford U.


00:43:31 6 Improving it with PyTorch and GPU, with Fastai Naive Bayes or ‘Fastai NBSVM++’ and “class DotProdNB(nn.Module):”
 Note: this long section includes lots of mathematical demonstration and explanation.


01:17:30 7 Deep Learning: Structured and Time-Series data with Rossmann Kaggle competition, with the 3rd winning solution ‘Entity Embeddings of Categorical Variables’ by Guo/Berkhahn.


01:21:30 1 Rossmann Kaggle: data cleaning & feature engineering.
 Using Pandas to join tables with ‘Left join’


Lesson 12
Note: you may want to pay specific attention to the second part of this final lesson, where Jeremy brings up delicate issues on Data Science & Ethics.
This goes beyond what most courses on DS cover.
00:00:01 13 Final lesson program !


00:01:01 1 Review of Rossmann Kaggle competition with ‘lesson3-rossman.ipynb’
 Using “df.apply(lambda x:…)” and “create_promo2since(x)”


00:04:30 1 Durations function “get_elapsed(fld, pre):” using “zip()”
 Check the notebook for detailed explanations.


00:16:10 3 Rolling function (or windowing function) for moving-average
 Hint: learn the Pandas API for Time-Series, it’s extremely diverse and powerful


00:21:40 1 Create Features, assign to ‘cat_vars’ and ‘contin_vars’
 ‘joined_samp’, ‘do_scale=True’, ‘mapper’,
 ‘yl = np.log(y)’ for RMSPE (Root Mean Squared Percent Error)
 Selecting a most recent Validation set in Time-Series, if possible of the exact same length as Test set.
 Then dropping the Validation set with ‘val_idx = [0]’ for final training of the model.


00:32:30 How to create our Deep Learning algorithm (or model), using ‘ColumnarModelData.from_data_frame()’
 Use the cardinality of each variable to decide how large to make its embeddings.
 Jeremy’s Golden Rule on difference between modern ML and old ML:
 “In old ML, we controlled complexity by reducing the number of parameters.
 In modern ML, we control it by regularization. We are not much concerned about Overfitting because we use increasing Dropout or Weight-Decay to avoid it”


00:39:20 1 Checking our submission vs Kaggle Public Leaderboard (not great), then Private Leaderboard (great!).
 Why Kaggle Public LB (LeaderBoard) is NOT a good replacement to your own Validation set.
 What is the relation between Kaggle Public LB and Private LB ?


00:44:15 4 Course review (lessons 1 to 12)
 Two ways to train a model: one by building a tree, one with SGD (Stochastic Gradient Descent)
 Reminder: Tree-building can be combined with Bagging (Random Forests) or Boosting (GBM)


00:46:15 3 How to represent Categorical variables with Decision Trees
 One-hot encoding a vector and its relation with embedding


00:55:50 1 Interpreting Decision Trees, Random Forests in particular, with Feature Importance.
 Use the same techniques to interpret Neural Networks, shuffling Features.


00:59:00 2 Why Jeremy usually doesn’t care about ‘Statistical Significant’ in ML, due to Data volume, but more about ‘Practical Significance’.


01:03:10 2 Jeremy talks about “The most important part in this course: Ethics and Data Science, it matters.”
 How does Machine Learning influence people’s behavior, and the responsibility that comes with it ?
 As a ML practicioner, you should care about the ethics and think about them BEFORE you are involved in one situation.
 BTW, you can end up in jail/prison as a techie doing “his job”.


01:08:15 2 IBM and the “Death’s Calculator” used in gas chamber by the Nazis.
 Facebook data science algorithm and the ethnic cleansing in Myanmar’s Rohingya crisis: the Myth of Neutral Platforms.
 Facebook lets advertisers exclude users by race enabled advertisers to reach “Jew Haters”.
 Your algorithm/model could be exploited by trolls, harassers, authoritarian governements for surveillance, for propaganda or disinformation.


01:16:45 Runaway feedback loops: when Recommendation Systems go bad.
 Social Network algorithms are distorting reality by boosting conspiracy theories.
 Runaway feedback loops in Predictive Policing: an algorithm biased by race and impacting Justice.


01:21:45 1 Bias in Image Software (Computer Vision), an example with Faceapp or Google Photos. The first International Beauty Contest judged by A.I.


01:25:15 1 Bias in Natural Language Processing (NLP)
 Another example with an A.I. built to help US Judicial system.
 Taser invests in A.I. and body-cameras to “anticipate criminal activity”.


01:34:30 2 Questions you should ask yourself when you work on A.I.
 You have options !



