# kmerPaPa

Tool to calculate a "k-mer pattern partition" from position specific k-mer counts. This can for instance be used to train a mutation rate model.

## Requirements

kmerPaPa requires Python 3.7 or above.

## Installation


## Usage
If we want to train a mutation rate model then the input data should specifiy the number of times each k-mer is observed mutated and unmutated. One option is to have one file with the mutated k-mer counts (positive) and one file with the count of k-mers in the whole genome (background).  We can then run kmerpapa like this:
```
kmerpapa --positive test_data/mutated_5mers.txt --background test_data/background_5mers.txt --penalty_values 3 5 7
```
The above command will first use cross validation to find the best penalty value between the values 3,5 and 7. Then it will find the optimal k-mer patter partiton using that penalty value.
If both a list of penalty values and a list of pseudo-counts are specified then all combinations of values will be tested during cross validation:
```
kmerpapa --positive test_data/mutated_5mers.txt --background test_data/background_5mers.txt --penalty_values 3 5 6 --pseudo_counts 0.5 1 10
```