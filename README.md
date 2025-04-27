# Statistical power and sample size estimation

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
The data were mined from figures using either dev_extract_points.py or ??,
or simply collected from tables. Extracted data are found as .csv in the folder: data/pubX

Replicated figures along results are found in the figures folders. 

**Publication 7**: replication of figures 6 and 7 (A + B) using extracted points.
Note that the pearson correlation coefficients vary somewhat to the original, some points were omitted,
but the p-values are representative. 

## Repeated subsampling to evaluate power
Replaced subsamples (bootstraps) of size n=3 to original N were repeatedly sampled 1000 times to evaluate the rate of
statistical significance under the reported success criteria (i.e. the statistical power).

If not reported in the study the type I error rate acceptance of p<.05 were used.

