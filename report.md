# Task 1 Financial Crisis Prediction

# Introduction
The dataset used for this task is from Schularick and Taylor(2012, "Credit Booms
Gone Bust"). It is an annual dataset covering 14 countries and 140 years. Among
the variables they collected, the most important one is the yearly aggregate
bank loans, which turn out to be the main soure of prediction power. The
variable of interest is the CrisisST which take a value of 1 when there is a
financial crisis and 0 otherwise. The financial crises are defined as "events
during which a country's banking sector experiences bank runs, sharp increases
in default rates accompanied by large losses of capital that result in public
intervention, bankrupcy, or forced merger of financial institutions"(Schularick
and Taylor, 2012).

Following the guidance of Schularick and Talor(2012), I explored the distinctive
relationships between several macro variables within two eras of finance
capitalism, tested the predicting power of different macro variables, and
compared predicting power of different supervised learning methods. The last
part of above is also inspired by the Fricke(2017, Financial Crisis Prediction:
A Model Comparison"). I managed to reproduce the results of two eras of finance
capitalism almost exactly as in Schularick and Taylor paper, and confirm their
results regarding the powerful explanatory variables. In the model comparison
part, my results alines with Fricke(2017) and shows that no model significantly
out perform logistic regression method.

The rest of this report is orgnized as follow: section 2 describes the basic
features of major variables and data cleaning method. Section 3 describes the
parameter choices of models and compares the prediction power via validation
methods. Section 4 conclude the report.

# Data describtion and cleaning
To explore the changing features between two eras, firstly I created variables
of interest: credit to GDP ratio, bank asset to GDP ratio, money to GDP ratio,
credit to money ratio, and bank asset to money ratio. To see the distinctive
trends in different historical periods, I regroup the whole dataset by years
and take mean value of each variable each year. Ploting mean values of the
ratios above against time, I recovered Figure 1 and 2 in Schularick and
Taylor(2012).

Figure 1 shows that bank loans, bank asset and broad money supply remain
basically steady related to the size of economy representing by GDP,
before the WW2 period. After the war, the money to GDP ratio stays flat while
the other two start to increase dramatically. This indicates a structure shift
in the post war economy. In the pre war period, money and credit are both tied
with GDP and grew hand by hand. Which means if some event can be explained by
credit then it is also possible to explain it by broad money supply. This
relationship was broken in the post war era and credit suplly start to grow
much faster than money supply. Figure 2 confirmed this observation from a
different angle. We can see that the loan to money ratio and bank asset to
money ratio start to take off after the distruction of WW2 implying that credit
start to grow faster than broad money supply and no steady relationship between
the too can be found in this period.

## put figure 1 and 2 here

To study financial crisis caused by the economy system, we need to exclude the
crisis caused by the two world war. As did in Schularick and Taylor paper, I
excluded the war period(1914 to 1919 and 1939 to 1947) and German crisis after
WW1(1920 to 1925). Divided the cleaned up dataset into pre- and post-war
period, I recovered the upper panel of table 1 in the paper.

# Annual summary statistics
# pre-war
 ;credit_to_GDP;bankAsset_to_GDP;money_to_GDP;credit_to_money;bank_asset_to_money
count;685;611;736;662;580
mean;0.408977;0.714051;0.533292;0.735337;1.282481
std;0.359888;0.447337;0.207534;0.449343;0.566104
# post-war
 ;credit_to_GDP;bankAsset_to_GDP;money_to_GDP;credit_to_money;bank_asset_to_money
count;831;828;834;833;831
mean;0.546975;1.013497;0.645801;0.838012;1.575839
std;0.423878;0.668770;0.240497;0.494226;0.752540

# Summary of delta log for critical variables
# pre-war
   ;loans1;bassets2;money;credit_to_money;bank_asset_to_money
mean;0.060365;0.054213;0.084300;0.009116;-0.000444
std;0.176997;0.161224;0.243966;0.086974;0.032686
# post-war
	;loans1;bassets2;money;credit_to_money;bank_asset_to_money
mean;0.128661;0.118789;0.109844;0.022129;0.017837
std;0.063803;0.055795;0.055891;0.042620;0.026840

# Table 1: this is a replication of table 1 in Schularick and Taylor paper

Defined delta(should be letter delta) log as the difference of log, which can
be interpreted as a growth rate or changing rate, I also tried to recover the
lower panel of table 1 in the paper, but the resulting numbers are slightly
different. Never the less, we can still see the facts mentioned in the paper
that money growth rate is significantly slower than credit growth rate in
post-war period.

Considering that more than half of the countries in the dataset are from
Europe, I also excluded all the observations in post WW1 period(1920 to 1925)
instead of only data from Germen since all the crises happened in that period
could be caused by WW1 rather than economical system. I also dropped any row
that contains missing value in any of these following six columns: year,
country, loans, cpi, credit to GDP ratio, crisisST. The choice of columns are
made based on Schularick and Taylor paper. I only kept the variables that have
protentially strong prediction power according to the results and the robustness
test from the paper.

After cleaning, I have 1433 observations and 59 out of these are crisis events.

# Supervised Learning Methods for classification

# Logistic regression and choice of explanory variables

The methods used in this part are logistic regression, classification tree,
classification forest, SVM and neural netwrok. The major criteria used here is
area under receiver operating curve. The secondary creteria is confusion
matrix. The major validation method is the modified cross validation method
mentioned in the Fricke paper that take time into consideration.

As defined in Schularick and Taylor paper, I created cpi nomalized bank loan
and take the difference of log value of this variable as the change of credit
environment. This variable will be called credit change in the rest of report.
I also take credit to GDP ratio as one protential explanatory variable since
in the robustness test of the paper, this variable seems to add prediction
power to the model. This variable lagged 1 year will be called credit size in
the remaining reprot. After assign each country for each year its lagged 1 to 5
credit change, I sorted the dataset by time to make sure that when fitting a
model, it is not trying to predict 1960s crisis with 1990s' data.

I started the analysis by useing lag 1 to lag 5 diff-log credit to fit a
logistic model. The auc is slightly higher than 0.5 for the whole dataset and
for the pre-war dataset, and significantly higher than 0.5 for the post-war
dataset. There is no strong evident indicate that the credit size add
predicting power to the model.

## AUC for logistic with lag 1 to 5 credit change

   ;whole set;pre-war;post-war
without credit size;0.5861360718870345;0.5762987012987013;0.7608543417366948
with credit size;0.5892169448010269;0.5800865800865801;0.615546218487395

> table 2: in-sample logistic regression trained with wholde time period,
> pre-war period and post war period with lag 1 to lag 5 credit change as major
> explanatory variable.

Since in the paper, lag 2 credit change is the only lagged variable that is
significant, I also fitted logistic regression with only lag 2 data and the
model fit slightly better for both whole set and pre-war period in respect of
auc. The change in post-war period is umbiguous. And credit size seems add
prediction power to the post-war period. From the following table we can see
that the lag 2 credit change is indeed the main sourece of information.

## AUC for logistic with lag 2 credit change

   ;whole set;pre-war;post-war
without credit size;0.6373277827336704;0.6286424526999033;0.697533908754624
with credit size;0.4518906730102092;0.5177922018137459;0.7200369913686806

> table 3: in-sample logistic regression trained with wholde time period,
> pre-war period and post war period with lag 2 credit change as major
> explanatory variable.

Here are plots of receiver operation curve of logistic regression for three time
period.

> Figure 3: AUC of logistic model fitted with whole time period, pre-war and
> post-war period. First two sets are fitted with lag 2 credit change only and
> the post-war period is also fitted with credit size.

I also did a simple out-of-sample test for the choice of variable using 30
percent of the dataset as test set and 70 percent as training set. The result
show that model fitted with lag 2 credit change have significantly better
out-of-sample performance than model fitted with lag 1 to lag 5 and credit size
still doesn't seem to add predicting power to the model. The auc are reported
in the following tables. And the plot of AUC is reported in figure below.

## AUC for logistic with lag 1 to lag 5 credit change

   ;whole set;pre-war;post-war
without credit size;0.5861360718870345;0.5762987012987013;0.7608543417366948
with credit size;0.5892169448010269;0.5800865800865801;0.615546218487395

> table 4: out-of-sample logistic regression trained with 70 percent of each data
> set and tested on 30 percent of data with lag 1 to lag 5 credit change as major
> explanatory variable.

## AUC for logistic with lag 2 credit change

   ;whole set;pre-war;post-war
without credit size;0.7370988446726572;0.7976190476190476;0.7461484593837535
with credit size;0.531193838254172;0.6737012987012987;0.6355042016806722

> table 5: out-of-sample logistic regression trained with 70 percent of each data
> set and tested on 30 percent of data with lag 2 credit change as major
> explanatory variable.

> Figure 4: out-of-sample AUC of logistic model fitted with 70 percent of data as
> training set and 30 percent as testing set. All three period fitted with
> lag 2 credit change only.

Given that lag 2 credit change is major source of predicting power and out
perform the other set of variable in the out-of-sample test, in the rest of
this study, I only considered lag 2 credit change and credit size variable.
This also means all the model compared here will have same information as
input and thus makes the comparison meaningful.

At last for logistic regression, I did a modified cross validation mentioned in
the Fricke paper. I divided the wholde dataset in to four equal folds. First,
I use fold one to train and test on fold two. Then I use fold one and two to
train and test on fold three, etc.. The average AUC is 0.56129 which is higher
than 0.5. The reason for divide the dataset into 4 rather than 5 fold like did
in the Fricke paper is that when the fold is too small, due to the sparseness
of crisis events, there might be only one class in the whole training set or
test set. This is also the reason for not be able to perform this test on pre-
and post war dataset.

# Tree and forest

Next I fitted the data with a classification tree. With maxmum depth equals to
3, here are result of in-sampel prediction. It is obvious from the table that
credit size add on predicting power for classification tree at least for
in-sample test.

## AUC for classification tree

   ;whole set;pre-war;post-war
without credit size;0.6937568143522649;0.7030106338903466;0.780209617755857
with credit size;0.7598684210526315;0.7878746029553929;0.8358199753390876

> table 6: in-sampel classification tree fitted with lag 2 credit change as major
> explanatory variable.

From the AUC plot we can see that there are much less point on each line for
the tree compared with logistic regression. This because for logistic
regression, each observation will have its own estimated probability, but for a
tree, the observation belong to the same leaf will have the same probability.

> Figure 5: AUC of classification tree fitted with whole time period, pre-war and
> post-war period. All three period fitted with lag 2 credit change and credit
> size.

For the choice of maxmum depth, we need a good balance between fully use all the
information and avoid over fitting. To find the maxmum depth that gives highest
AUC for each tree, I performed an analysis with following steps:
1> for each of whole set, pre-war, and post-war period, set up a modified cross
validation of 5 fold as mentioned in logistic regression part with certain
maxmum depth and record the average AUC for each model.
2> collect average AUC of each model for maxmum depth from 2 to 50.
3> plot the average AUC against maxmum depth for each model and pick up the
depth that coincide with the highest AUC.

> Figure 6: Average AUC for the whole, pre- and post-war dataset fitted with and
> without credit size plot against maxmum depth

The result show that the optimal depth is around 5 for all the models. With
optimal max-depth 3, 3, 5 respectlly, I fitted whole period and pre-war dataset
with lag 2 credit change and post-war with lag 2 credit change and credit size.
With a test set of the size of 30 percent of the total data, model fitted with
with three datasets show AUC 0.62895, 0.69021 and 0.43697 respectlly.

The dramatically different performance between in-sample and out-of-sample test
is caused by over fitting. Given a high enough max-depth, classification tree
can acheive AUC 1.0 in an in-sample test. But over fitting will lead to very
poor out-of-sample prediction. This is indicated by the flatening out in the
figure above. I also plotted two trees with different maxmum depth in the
appendex to demonstrate this point.

With exactly same idea, I performed analysis with classification forest model.
The in-sample performance is better than the tree with same max-depth which
could either caused by higher predicting power or overfitting.

## AUC for classification forest

   ;whole set;pre-war;post-war
without credit size;0.7419033105362275;0.7823735211526954;0.9169852034525278
with credit size;0.7619994548518189;0.8678359342632234;0.8669852034525277

> table 6: in-sampel classification forest fitted with lag 2 credit change as major
> explanatory variable.

> Figure 7: AUC of classification forest fitted with whole time period, pre-war and
> post-war period. First two period fitted with lag 2 credit change and credit
> size and post-war fitted with lag 2 credit change only.

With optimal max-depth picked up by the same method as in the tree analysis,
datasets are fitted with lag 2 credit change and/or credit size. Tested with 30
percent data in the dataset, the AUC are 0.62426, 0.73593, and 0.71709 for
whole, pre-war and post-war data.

The result from classificaton tree and forest alined with the result from
Fricke paper. These two methods perform brilliantly in-sample and are not much
better than random guess in out-of-sample test. Forest perform slightly better
than tree method.

In both tree and forest analysis, I used gini index and entropy and there are
little difference between the AUC.

# A few interesting facts and protential explanation

When taking a closer look to the results of tree and forest, I found a few
interesting facts. First, for both tree and forest, I found the results are
different among different function calls with exactly same parameters. This
indecats that there is randomness involded in the fitting precesure. I belive
this is randomness rather than unstableness because no parameter or dataset was
changed at all. Second, for the model fitted with both lag 2 credit change and
credit size, the average AUC show fluctuation in the over fitting tail. In the
plot of forest, all model show fluctuation in the over fitting tail parts but
for models with two explainory variables, the fluctuations have higher
amplitude. Last but not least, for a few models, both in tree and forest, the
out of sample AUC do not drop to 0.5 even in the obviously over fitting zone.

To answer the first question, one need to identify the souce of randomness
which might be multiple. After studying of the document of the sklearn, I
found two protential sources for tree and three for forest. One obvious
candidate is the random start point when searching for the optimal arguments to
minimize cost function. However, as long as the problem is convex, the
searching reslults should still be with in a relatively small intervel since
they should all be close to one true optimal. The difference might be caused by
limited resolution. This doesn't match the jumps I observed. Another protential
is an argument called max_features in tree: "The number of features to consider
when looking for the best split(sklearn.tree.DecisionTreeClassifier document
page)". This does not make difference when the model was fitted with only one
variable. When fitted with two variables, the model with max feature setted to
1 shows jumps among fittings with exactly same parameters. The medel with max
features setted to 2 does not. This explained the source of randomness of the
tree method. To satisfy the limitation on max fearture number, the randomness
is involved in the searching rule. To understand how exactly the rule was set,
one need to dive into the source code of the library. A few trees generated by
same dataset are included in the appendex. For the forest, there is one extra
source of randomness. To grow multiple trees, the sklearn library bootstrap
observations using random selection with replacement. To test this conclusion,
I fit the model with same dataset multiple times and found that the jumps only
disappare when you lift up limitation on max fearture and turn off bootstrap.
If either one of the two was still on, their will be jumps among the results.
In both tree and forest sklean function, their is a argument called random
state that controls all the randomness. This means fix this argument will turn
off all the randomness in the model.

Carrying knowledge mentioned above, I try to decompose the fluctuations. First,
I fix random state argument, this eliminate fluctuation in both AUC-max-depth
plot in the tail part which means all the fluctuations are caused by
randomness. Next, I generate the AUC-max-depth plot with no limit on max
feature, and observed that the shape of the over fitting part changed but the
amplitudes of AUC are not obvious reduced. This indicate that the limit on max
feature is not the main source of observed fluctuation and there are other
usage of randomness in the model fitting that are unknown. For the forest, I
first turn off the max feature limit, and the resulting plot doesn't show less
fluctuation. Next, I add the limit on max feature back and turn off bootstrap,
and this reduced the fluctuation. And eventually, when I turn off both limit
and bootstrap, the fluctuation was almost totally eliminated. These changes
indicates that most of the fluctuation is because that the forest fitting is
extremely sensitive to the chang of training dataset. Another interesting
observation here is that the AUC of model fitted with pre-war data and with
both credit change and credit size as explainory variables dropped below 0.5
benchmark after the limit and bootstrap was turn off. I cannot explain this
change.

> Figure 8: The maxmum feature is limited and bootstrap is on, this is exactly the
> same plot used to choose optimal depth. We can see a lot of fluctuation here
> at least three AUC do not approach 0.5 benchmark.

> Figure 9: The maxmum feature is limited and bootstrap is off, and we can see
> that fluctuation is reduced.

> Figure 10: The maxmum feature is unlimited and bootstrap is off, and we can see
> dramatical reduce of fluctuation and only two lines do not approach 0.5
> benchmark.

To answer the last question, first, I notice that the model that has prediction
power even when over fitted are models fitted with the whole dataset. This
observation coincide in both tree and forest analysis(with randomness turn off
in forest). From the plot of AUC of trees fitted with different datasets, I
notice that there is only one point in each of the three line. This indecates
that in the out of sample test, only the first split take effect and did
actual spliting. Other split point, most likely due to over fitting, do not take
effect. Considering that the whole set has the largest number of observations, it
is likely that even when over fitted, the first split has some predicting power
and thus the model show AUC higher than 0.5 in the out of sample test. Here is
one exampel of AUC plot when over fitted. This also explains the same
observation in the forest analysis.

> Figure 11: All three models are fitted with lag 2 credit change and credit size. These
> models are fitted with 80 percent of dataset and tested on 20 percent of
> set. To eliminate effect of randomness, the ramdom state argument was fixed
> when fit these models.

# Another Criteria

Except for AUC, confusion matrix is also a good criteria for model comparison.
In the tables below, I collected results from logistic regression and forest.
Here are some definations and interprations:
Threshold: Model output value used as threshold.
Sensitiviti: M11 / (M11 + M10), percentage of really crisis that are captured
by model.
False alarm: M01 / (M01 + M11), when model say crisis, how much of that are
miss classified.
Total flag: (M01 + M11) / test observation, percentage of total observation
that has been flagged by the model as crisis.

# Out-of-sampel performance for logistic regression based on confusion matrix

threshold;sensitiveity;falseAlarm;totalFlag
0.9584492738912054;0.05263157894736842;0.875;0.018648018648018648
0.9584156227543202;0.10526315789473684;0.8;0.023310023310023312
0.9582944101957735;0.15789473684210525;0.8;0.03496503496503497
0.9581720526209747;0.21052631578947367;0.84;0.05827505827505827
0.9581434201492378;0.2631578947368421;0.8333333333333334;0.06993006993006994
0.9580476223194564;0.3157894736842105;0.8604651162790697;0.10023310023310024
0.9580060539159138;0.3684210526315789;0.86;0.11655011655011654
0.9579924986498776;0.42105263157894735;0.8518518518518519;0.1258741258741259
0.9579584325457295;0.47368421052631576;0.8571428571428571;0.14685314685314685
0.9579263005580301;0.5263157894736842;0.8717948717948718;0.18181818181818182
0.9579072932281031;0.5789473684210527;0.8817204301075269;0.21678321678321677
0.9578986698802621;0.631578947368421;0.8811881188118812;0.23543123543123542
0.9578854148873172;0.6842105263157895;0.8828828828828829;0.25874125874125875
0.9578553847205481;0.7368421052631579;0.8931297709923665;0.30536130536130535
0.9577269139408369;0.7894736842105263;0.9282296650717703;0.48717948717948717
0.9576669020322022;0.8421052631578947;0.937007874015748;0.5920745920745921
0.95763518803743;0.8947368421052632;0.9388489208633094;0.6480186480186481
0.9575910472738063;0.9473684210526315;0.940983606557377;0.710955710955711
0.9574512307806254;1.0;0.95;0.8857808857808858

> Table:Logistic regression fitted with 70 percent of the whole period and
> tested with 30 percent of whole period. Explanory variable is lag 2 credit
> change only. AUC is 0.73709.

# Out-of-sampel performance for forest based on confusion matrix

threshold;sensitiveity;falseAlarm;totalFlag
0.31087996870780027;0.05263157894736842;0.0;0.002331002331002331
0.15586808878177918;0.10526315789473684;0.5;0.009324009324009324
0.0538083675328781;0.15789473684210525;0.9090909090909091;0.07692307692307693
0.047849724485387976;0.3684210526315789;0.8939393939393939;0.15384615384615385
0.04160062144965254;0.7368421052631579;0.9366515837104072;0.5151515151515151
0.03808155845225508;0.8421052631578947;0.9382239382239382;0.6037296037296037
0.03176986812156413;0.8947368421052632;0.9388489208633094;0.6480186480186481
0.010696363086234579;0.9473684210526315;0.95;0.8391608391608392
0.00829026056619624;1.0;0.9557109557109557;1.0

> Table: Forest fitted with 70 percent of the whole period and tested on 30
> percent of the whole period. Variable is lag 2 credit change only, max depth
> is 3. The AUC of this model is 0.67079.

We can see from these two tables that to capture 10% of the crisis, logistic
regression need to flag 2% of the total observation, and 80% of those flags are
false alarm. On the other hand, forest need only flag less than 1% of data and
only half of those are false alarm. However, if the goal is to capture half of
the crisis, logistic regression only need to flag less than 20% of data with
87% false alarm while forest need to flag half of the data point and more than
90% of the flag are false alarm. The main idea here is that there is no sigle
best criteria and the criteria selection should be based on the goal of
analysis.

# SVM

When fitting SVM model with the data, I used two kinds of kernels: rbf(Gaussion
kernel) and sigmoid kernel. The results are both affected by randomness when
fitting the model and sigmoid kernel in general has better out of sample
perfomance. In the table below is the result of this out of sample test. Seems
this method is especially good at pre-war period fitted with lag 2 credit
change.

# Out-of-sampel AUC for SVM with sigmoid kernel

   ;whole set;pre-war;post-war
without credit size;0.6340571947668452;0.7898343285048894;0.5956201469707326
with credit size;0.5751033061514164;0.6282002404217119;0.6295270129981603

> table: Mean AUC for models under 5 fold modified cross validation.

# Conclusion and protential further questions to answer

In this study, I explored the different features of a credit environment before
and after WW2 and confirm the result from Schularick and Taylor paper. Bank
loan in major developed countries use to grow together with GDP and broad money
supply before WW2. In the post war period, however, it start to grow in a speed
much faster than those two. To confrim that lagged credit change contains
information to predict crisis, I implement a few supervised learning models on
the panel data. The major predicting power lies in lag 2 credit change and this
conclusion alines with Schularick and Taylor paper. In respect of supervised
models, their power are much stronger in sample. This confirms the result of
Fricke paper. Among the models I fitted, tree and forest are particularly
vulnarable to over fitting thus require carefully picked parameters like maxmum
depth or maxmum number of leaves. They are also quite sensitive to the change
of training set. This is explored by truning on and off the bootstrap on
observations when training forest. Another issue for tree and forest is that
when there is a dominating class in the training data, there model generated
could be biasd. This issue can be reduced by balance the data prior fitting or
assign similar weight to different classes. However, these are not explored in
this report and can be interesting topic for futher study. Since forest shows
great protential to predict crisis, and machine learning methods are not widly
use in economical research, solving this issue may have practical meaning.
All the models tested in this report show some level of predicting power.
However, there is no sigle standard to judge which is the best model. The
selecting criteria strongly depends on the purpose of the analysis and
different models may have adventages in different tasks.

# Reference

[1] Credit Booms Gone Bust: Monetary Policy, Leverage Cycles, and Financial
Crisis, 1870-2008 by Moritz Schularick an dAlan M. Taylor
[2] Financial Crisis Prediction: A Model Comparison by Daniel Fricke
[3] sklearn document
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Appendix

# app_normaldepth.pdf
Figure 1: nomal depth tree
# app_overfitting.pdf
Figure 2: over fitting tree
# app_samedata_1.pdf
# app_samedata_2.pdf
Figure 3: different trees generated by same data with limited max feature
