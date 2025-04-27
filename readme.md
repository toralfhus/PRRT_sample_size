# Statistical power and sample size estimation using a monte carlo approach

Using data from publications evaluating the dose-response between absorbed tumour dose (AD) and radiological response 
after 177Lu-Dotatate [peptide-receptor radionuclide therapy (PRRT)](https://en.wikipedia.org/wiki/Peptide_receptor_radionuclide_therapy) of gastroenteropancreatic neuroendocrine tumours (GEP-NETs).

10 publications were considered (not finished):
1. x
2. x
3. x
4. x
5. x
6. x
7. Jahn et al, 2021, Cancers, https://doi.org/10.3390/cancers13050962
8. x
9. x
10. x


## Replication of data
The data were mined from figures, or simply collected from tables. Extracted data are found as .csv in the folder 
data/pubX and loaded using functions in utils_load.py. Replicated figures along results are found in the figures folders.

**Publication 7 as example**: Running the script [pub7.py](pub7.py)

- First a generator of extracted points from the four figures are loaded using the function **load_pub7(folder_data)** from [utils_load.py](utils_load.py)
- Figure 6A is replicated using extracted points. 

<img src="./figures/pub7_jahn21/readme_data.png" alt="alt text" width="800"/>

Note that the pearson correlation coefficient vary somewhat to the original (R2 of 0.33 versus 0.37), but the p-value is similar.


## Repeated subsampling to evaluate power
Replaced subsamples (bootstraps) of size n=3 to original N were repeatedly sampled 1000 times to evaluate the rate of
statistical significance under the reported success criteria (statistical power), i.e. a monte-carlo approach. If the success criterie were not reported in the study the type I error rate acceptance of p < .05 were used.

Publication 7 uses a double criteria: both $R^2 \geq 0.25$ and p < .05
<br>

<img src="./figures/pub7_jahn21/readme_power.png" alt="alt text" width="300"/>

Evaluating the figure, it seems to converge to a power about